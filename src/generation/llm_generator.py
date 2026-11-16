"""
LLM Generator — Generation Layer
====================================
Default backend loads TinyLlama locally with Hugging Face Transformers.
No Ollama server and no cloud API.

Optional backend "ollama" still calls a local Ollama server at
http://localhost:11434/api/generate.

If the chosen backend cannot generate, the module returns an extractive
snippet from the top retrieved chunk so detection can still be tested.
"""

import requests


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, factual assistant that ONLY answers based on the provided context.

Rules:
1. Answer ONLY using information from the context below.
2. If the context does not contain enough information, say: "The provided documents do not contain sufficient evidence to answer this question."
3. Do NOT fabricate information, statistics, or references not present in the context.
4. Be concise and factual.
5. Write complete sentences."""


def _plain_excerpt(text: str, max_chars: int = 700) -> str:
    """Collapse a retrieved chunk to one paragraph, clipped at a sentence end."""
    passage = " ".join((text or "").split()).lstrip("# ").strip()
    if len(passage) <= max_chars:
        return passage
    cut = passage[:max_chars]
    dot = cut.rfind(". ")
    if dot > 200:
        return cut[: dot + 1]
    return cut.rstrip() + "..."


def build_prompt(query: str, context_chunks: list) -> str:
    """
    Build the full LLM prompt from the user query and retrieved chunks.

    Args:
        query: The user's question.
        context_chunks: List of retrieved chunk dicts (with 'text', 'source_file', 'page_num').

    Returns:
        Formatted prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_num", "?")
        score = chunk.get("similarity_score", 0.0)
        context_parts.append(
            f"[Context {i+1}] Source: {source}, Page: {page} (similarity: {score:.3f})\n"
            f"{chunk['text']}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

=== CONTEXT ===
{context_text}

=== QUESTION ===
{query}

=== ANSWER ==="""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Generator class
# ─────────────────────────────────────────────────────────────────────────────

TINYLLAMA_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_CHUNKS = 3
TINYLLAMA_WORDS_PER_CHUNK = 280


class LLMGenerator:
    """
    Generates a grounded answer from retrieved chunks.

    backend="tinyllama" loads TinyLlama in this process.
    backend="ollama" calls a local Ollama server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = TINYLLAMA_ID,
        temperature: float = 0.1,
        max_tokens: int = 400,
        backend: str = "tinyllama",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name or TINYLLAMA_ID
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = (backend or "tinyllama").lower()
        self._model = None
        self._tokenizer = None
        self._device = None
        if self.backend == "ollama":
            self._check_connection()
        else:
            self.backend = "tinyllama"
            if "tinyllama" not in self.model_name.lower():
                self.model_name = TINYLLAMA_ID
            self._fallback_mode = False
            print(f"[INFO] Generation backend: TinyLlama ({self.model_name}). "
                  "The weights load on the first question.")

    def _check_connection(self) -> None:
        """Check if Ollama server is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                print(f"[INFO] Ollama running. Available models: {models}")
                if self.model_name not in " ".join(models):
                    print(f"[WARN] Model '{self.model_name}' may not be pulled. "
                          f"Run: ollama pull {self.model_name}")
            else:
                print(f"[WARN] Ollama responded with status {resp.status_code}")
        except Exception:
            print(f"[WARN] Ollama not reachable at {self.base_url}. "
                  "Fallback mode enabled (template responses only).")
            self._fallback_mode = True
            return
        self._fallback_mode = False

    def generate(
        self,
        query: str,
        context_chunks: list,
    ) -> dict:
        """
        Generate an answer for the given query using retrieved context chunks.

        Args:
            query: User question string.
            context_chunks: List of retrieved chunk dicts from FAISS search.

        Returns:
            dict with keys:
              - "answer": Generated text string.
              - "prompt": Full prompt sent to LLM.
              - "model": Model name used.
              - "fallback": True if the model could not generate.
        """
        prompt = build_prompt(query, context_chunks)

        if self.backend == "tinyllama":
            return self._generate_tinyllama(query, context_chunks, prompt)

        if getattr(self, "_fallback_mode", False):
            return self._fallback_response(query, context_chunks, prompt, "Ollama not running")

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stop": ["=== QUESTION ===", "=== CONTEXT ==="],
                },
            }

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "").strip()

            return {
                "answer": answer,
                "prompt": prompt,
                "model": self.model_name,
                "fallback": False,
            }

        except requests.exceptions.Timeout:
            print("[WARN] Ollama request timed out. Using fallback.")
            return self._fallback_response(query, context_chunks, prompt, "Ollama timed out")
        except Exception as e:
            print(f"[WARN] LLM generation error: {e}. Using fallback.")
            return self._fallback_response(query, context_chunks, prompt, "Ollama request failed")

    def warmup(self) -> bool:
        """Load the local model ahead of the first question. Returns True if it is ready."""
        if self.backend != "tinyllama":
            return not getattr(self, "_fallback_mode", False)
        if getattr(self, "_tinyllama_unavailable", False):
            return False
        try:
            self._load_tinyllama()
            return True
        except Exception as e:
            self._tinyllama_unavailable = True
            print(f"[WARN] TinyLlama could not be loaded: {e}")
            return False

    @property
    def status(self) -> str:
        """Short status for the UI."""
        if self.backend == "tinyllama":
            if self._model is not None:
                return f"TinyLlama ready on {self._device}"
            if getattr(self, "_tinyllama_unavailable", False):
                return "TinyLlama not available (using retrieved passage)"
            return "TinyLlama not loaded yet"
        if getattr(self, "_fallback_mode", False):
            return "Ollama not reachable (using retrieved passage)"
        return f"Ollama: {self.model_name}"

    def _load_tinyllama(self) -> None:
        """Download (once) and load TinyLlama onto MPS, CUDA, or CPU."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[INFO] Loading TinyLlama: {self.model_name}")
        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        elif torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype, local_files_only=True
        )
        model.to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        print(f"[INFO] TinyLlama ready on {device}.")

    def _generate_tinyllama(self, query: str, context_chunks: list, prompt: str) -> dict:
        """Generate with the local TinyLlama chat model."""
        if getattr(self, "_tinyllama_unavailable", False):
            return self._fallback_response(
                query, context_chunks, prompt, "TinyLlama is not available locally"
            )
        try:
            import torch

            self._load_tinyllama()
            # TinyLlama reads at most 2048 tokens, prompt and answer together.
            short_chunks = [
                dict(c, text=" ".join(_plain_excerpt(c.get("text", ""), 10_000).split()[:TINYLLAMA_WORDS_PER_CHUNK]))
                for c in context_chunks[:TINYLLAMA_CHUNKS]
            ]
            short_prompt = build_prompt(query, short_chunks)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": short_prompt.split("=== CONTEXT ===", 1)[-1].strip()},
            ]
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self._device)
            prompt_len = inputs["input_ids"].shape[-1]

            do_sample = self.temperature > 0
            gen_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": do_sample,
                "repetition_penalty": 1.15,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = self.temperature
            with torch.no_grad():
                output = self._model.generate(**inputs, **gen_kwargs)
            answer = self._tokenizer.decode(
                output[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()
            if not answer:
                return self._fallback_response(
                    query, context_chunks, prompt, "TinyLlama returned an empty answer"
                )
            return {
                "answer": answer,
                "prompt": prompt,
                "model": self.model_name,
                "fallback": False,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            if self._model is None:
                self._tinyllama_unavailable = True
            print(f"[WARN] TinyLlama generation failed: {type(e).__name__}: {e}")
            return self._fallback_response(query, context_chunks, prompt, f"TinyLlama failed: {e}")

    def _fallback_response(
        self,
        query: str,
        context_chunks: list,
        prompt: str,
        reason: str = "generation unavailable",
    ) -> dict:
        """
        Use the top retrieved passage itself when the generator cannot run.

        The error text is logged, not inserted into the answer. Markdown
        heading marks are removed so the passage stays normal sentence text.
        """
        print(f"[WARN] Using retrieved passage because generation failed: {reason}")
        if context_chunks:
            answer = _plain_excerpt(context_chunks[0].get("text", ""))
        else:
            answer = "No relevant documents found for this query."

        return {
            "answer": answer,
            "prompt": prompt,
            "model": "fallback",
            "fallback": True,
        }


if __name__ == "__main__":
    gen = LLMGenerator()
    fake_chunks = [
        {
            "text": "RAG combines retrieval with generation to produce grounded answers.",
            "source_file": "rag_paper.pdf",
            "page_num": 1,
            "similarity_score": 0.92,
        }
    ]
    result = gen.generate("What is RAG?", fake_chunks)
    print("\n=== Generated Answer ===")
    print(result["answer"])
    print(f"\nModel: {result['model']} | Fallback: {result['fallback']}")
