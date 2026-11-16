"""
Streamlit UI — TrustRAG conversational dashboard
================================================
Chat interface for the explainability-driven hallucination detection framework.

  • Models and the FAISS index load once per server (st.cache_resource), not per tab.
  • Each answer is colour-coded per sentence (green supported, amber weak, red hallucinated).
  • Explainability charts render only when a message's details are opened.
  • Chat history is saved to data/chat_history.pkl and restored on restart.
"""

import html
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Use the models already downloaded. Do not contact Hugging Face on each question.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HISTORY_FILE = Path(PROJECT_ROOT) / "data" / "chat_history.pkl"

GREETING = (
    "Hello! I answer questions from the documents in `data/corpus/`. "
    "Every sentence I write is checked against the retrieved evidence: supported sentences "
    "are cited, weak ones are flagged, and unsupported answers are blocked."
)

EXAMPLES = [
    ("✅ Cited", "What is BERTScore?"),
    ("⚠️ Partial", "What is Retrieval-Augmented Generation?"),
    ("🚫 Hallucination caught", "What is FAISS?"),
    ("🚫 Out of scope", "Who won the 2022 FIFA World Cup?"),
]

VERDICT_COLORS = {"SUPPORTED": "#22c55e", "WEAKLY_SUPPORTED": "#f59e0b", "HALLUCINATED": "#ef4444"}

# ─────────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="TrustRAG", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: #0E1117; color: #E5E7EB; }
    .block-container { padding-top: 1.6rem; max-width: 1100px; }

    .header { text-align: center; padding: 0.6rem 0 1.2rem 0; margin-bottom: 1.2rem;
              border-bottom: 1px solid rgba(255,255,255,0.08); }
    .title { background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; }
    .subtitle { color: #94a3b8; font-size: 0.92rem; margin-top: 0.15rem; }

    .answer { font-size: 1rem; line-height: 1.75; }
    .sent-supported { background: rgba(34,197,94,0.10); border-bottom: 2px solid #22c55e;
                      padding: 1px 4px; border-radius: 4px; }
    .sent-weak { background: rgba(245,158,11,0.10); border-bottom: 2px solid #f59e0b;
                 padding: 1px 4px; border-radius: 4px; }
    .sent-hallucinated { background: rgba(239,68,68,0.10); border-bottom: 2px solid #ef4444;
                         padding: 1px 4px; border-radius: 4px; color: #fca5a5;
                         text-decoration: line-through; }
    .cite { color: #93c5fd; font-size: 0.8rem; margin-left: 2px; }

    .badge { font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
             display: inline-block; margin: 0 6px 8px 0; }
    .badge-cite  { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.45); }
    .badge-warn  { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.45); }
    .badge-block { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.45); }
    .badge-info  { background: rgba(148,163,184,0.12); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.35); }

    .evidence { font-size: 0.92rem; line-height: 1.6; padding: 0.6rem 0.8rem; margin: 0.4rem 0;
                border-left: 3px solid #6366f1; background: #151a22; border-radius: 4px; }
    .evidence-head { color: #93c5fd; font-size: 0.82rem; margin-bottom: 0.25rem; }

    .status-row { font-size: 0.85rem; margin: 0.15rem 0; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .dot-ok { background: #22c55e; } .dot-warn { background: #f59e0b; } .dot-bad { background: #ef4444; }

    .saliency { line-height: 2.0; padding: 10px 12px; border-radius: 5px; background: #151a22;
                border-left: 3px solid #6366f1; margin-bottom: 10px; }

    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (loaded once per server process)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline import RAGHallucinationPipeline
    pipe = RAGHallucinationPipeline()
    pipe._load_models()
    stats = pipe.index_documents()          # reuses data/index if it exists
    pipe.generator.warmup()
    return pipe, stats


def _plain_text(text: str) -> str:
    """Document text as one escaped paragraph, so Markdown marks never render as headings."""
    return html.escape(" ".join((text or "").split()).lstrip("# "))


# ─────────────────────────────────────────────────────────────────────────────
# Chat history persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "rb") as f:
                messages = pickle.load(f)
            if isinstance(messages, list) and messages:
                return messages
        except Exception as e:
            print(f"[WARN] Could not read chat history ({e}). Starting a new chat.")
    return [{"role": "assistant", "content": GREETING, "result": None}]


def save_history():
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(st.session_state.messages, f)
    except Exception as e:
        print(f"[WARN] Could not save chat history: {e}")


if "messages" not in st.session_state:
    st.session_state.messages = load_history()
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ─────────────────────────────────────────────────────────────────────────────
# Load models (first run shows a progress message)
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading models and the document index (first start only)..."):
    pipe, index_stats = get_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛡️ System status")

    n_chunks = pipe.faiss_store.num_chunks if pipe.faiss_store.is_loaded() else 0
    gen_status = pipe.generator.status
    gen_ok = gen_status.startswith("TinyLlama ready") or gen_status.startswith("Ollama:")
    rows = [
        ("ok" if n_chunks else "bad", f"Index: {n_chunks} chunks"),
        ("ok" if gen_ok else "warn", f"Generator: {gen_status}"),
        ("ok", f"NLI: {pipe.nli_model.split('/')[-1]}"),
        ("ok", f"BERTScore: {pipe.bs_model}"),
    ]
    st.markdown("".join(
        f'<div class="status-row"><span class="dot dot-{s}"></span>{html.escape(t)}</div>'
        for s, t in rows
    ), unsafe_allow_html=True)

    if st.button("🔨 Rebuild document index", width="stretch"):
        with st.spinner("Re-indexing data/corpus ..."):
            try:
                stats = pipe.index_documents(force_reindex=True)
                st.success(f"Indexed {stats.get('num_chunks', '?')} chunks.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.markdown("---")
    with st.expander("🎛️ Detection tuning", expanded=False):
        w_nli = st.slider("NLI weight", 0.0, 1.0, float(pipe.w_nli), 0.05)
        w_bert = st.slider("BERTScore weight", 0.0, 1.0, float(pipe.w_bert), 0.05)
        w_cos = st.slider("Cosine weight", 0.0, 1.0, float(pipe.w_cos), 0.05)
        high_threshold = st.slider("Cite at or above", 0.0, 1.0, float(pipe.high_thr), 0.05)
        medium_threshold = st.slider("Warn at or above", 0.0, 1.0, float(pipe.medium_thr), 0.05)
        w_total = w_nli + w_bert + w_cos
        if w_total <= 0:
            w_nli, w_bert, w_cos, w_total = float(pipe.w_nli), float(pipe.w_bert), float(pipe.w_cos), 1.0
            st.caption("All weights are zero; using the configured weights.")
        elif abs(w_total - 1.0) > 1e-6:
            st.caption(f"Weights sum to {w_total:.2f}; they are rescaled to sum to 1.")
        w_nli, w_bert, w_cos = w_nli / w_total, w_bert / w_total, w_cos / w_total

    if st.button("🗑️ Clear chat history", width="stretch"):
        st.session_state.messages = [{"role": "assistant", "content": GREETING, "result": None}]
        HISTORY_FILE.unlink(missing_ok=True)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Explainability panel (rendered only when opened)
# ─────────────────────────────────────────────────────────────────────────────

def render_explainability_panel(result, key_suffix):
    tab_claims, tab_charts, tab_saliency, tab_chunks = st.tabs(
        ["Per-claim log", "Charts", "Word saliency", "Retrieved chunks"]
    )

    with tab_claims:
        if result.verdicts:
            st.dataframe(pd.DataFrame([{
                "#": v.sentence_idx + 1,
                "Claim": v.sentence,
                "NLI": round(v.nli_score, 3),
                "BERTScore": round(v.bertscore_f1, 3),
                "Cosine": round(v.cosine_score, 3),
                "Score": round(v.support_score, 3),
                "Verdict": v.classification,
            } for v in result.verdicts]), width="stretch", hide_index=True)

        export = {
            "query": result.query,
            "raw_answer": result.raw_answer,
            "state": result.response.state if result.response else None,
            "overall_support_score": result.summary.get("overall_support_score"),
            "provenance_score": result.xai_report.get("provenance", {}).get("provenance_score"),
            "latency_seconds": result.summary.get("latency"),
            "verdicts": [v.to_dict() for v in result.verdicts] if result.verdicts else [],
            "retrieved_chunks": result.retrieved_chunks,
        }
        st.download_button("📥 Export audit JSON", data=json.dumps(export, indent=2, default=str),
                           file_name=f"audit_report_{key_suffix}.json", mime="application/json",
                           key=f"audit_download_{key_suffix}")

    with tab_charts:
        if result.verdicts:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                for v in result.verdicts:
                    fig.add_trace(go.Scatterpolar(
                        r=[v.nli_score, v.bertscore_f1, v.cosine_score, v.support_score, v.nli_score],
                        theta=["NLI", "BERTScore", "Cosine", "Score", "NLI"],
                        name=f"Claim {v.sentence_idx + 1}",
                        line=dict(color=VERDICT_COLORS.get(v.classification, "#64748b")),
                        fill="toself", opacity=0.2,
                    ))
                fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1])),
                                  paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                                  height=280, margin=dict(l=40, r=40, t=20, b=20))
                st.plotly_chart(fig, width="stretch", key=f"radar_{key_suffix}")
            with c2:
                fig = go.Figure(go.Bar(
                    x=[v.support_score for v in result.verdicts],
                    y=[f"Claim {v.sentence_idx + 1}" for v in result.verdicts],
                    orientation="h",
                    marker_color=[VERDICT_COLORS.get(v.classification, "#64748b") for v in result.verdicts],
                    text=[f"{v.support_score:.2f}" for v in result.verdicts], textposition="auto",
                    hovertext=[v.sentence for v in result.verdicts], hoverinfo="text+x",
                ))
                for thr, color in ((pipe.controller.high_thr, "#22c55e"), (pipe.controller.medium_thr, "#f59e0b")):
                    fig.add_vline(x=thr, line_dash="dot", line_color=color)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(title="Support score", range=[0, 1.05]),
                                  yaxis=dict(autorange="reversed"), font=dict(color="#94a3b8"),
                                  height=280, margin=dict(l=60, r=20, t=20, b=20))
                st.plotly_chart(fig, width="stretch", key=f"bar_{key_suffix}")

    with tab_saliency:
        st.caption("Green words anchor the sentence to its evidence; red words pull it away "
                   "(leave-one-out embedding saliency).")
        for attr in result.xai_report.get("attributions", []):
            parts = []
            for w in attr.get("word_saliency", []):
                sal = w.get("saliency", 0.0)
                word = html.escape(w.get("word", ""))
                if sal > 0.002:
                    style = f"background: rgba(34,197,94,{min(0.65, 0.08 + sal * 4):.2f}); border-radius:3px; padding:1px 3px;"
                elif sal < -0.002:
                    style = f"background: rgba(239,68,68,{min(0.65, 0.08 + abs(sal) * 4):.2f}); border-radius:3px; padding:1px 3px;"
                else:
                    style = "color:#94a3b8; padding:1px 3px;"
                parts.append(f'<span style="{style}">{word}</span>')
            if parts:
                st.markdown(f"**Claim {attr.get('sentence_idx', 0) + 1}**")
                st.markdown(f'<div class="saliency">{" ".join(parts)}</div>', unsafe_allow_html=True)

    with tab_chunks:
        for i, chunk in enumerate(result.retrieved_chunks or []):
            st.markdown(
                f'<div class="evidence"><div class="evidence-head">#{i + 1} · '
                f'{html.escape(str(chunk.get("source_file", "?")))}, p.{html.escape(str(chunk.get("page_num", "?")))} '
                f'· similarity {float(chunk.get("similarity_score", 0.0)):.3f}</div>'
                f'{_plain_text(chunk.get("text", ""))}</div>',
                unsafe_allow_html=True,
            )


def render_answer(res, idx):
    resp = res.response
    fallback = getattr(resp, "generation_fallback", False)
    score = res.summary.get("overall_support_score", 0.0)
    n_hal = res.summary.get("num_hallucinated", 0)
    total = res.summary.get("total_sentences", 0)
    latency = res.summary.get("latency", 0.0)

    badge = {
        "ANSWER_WITH_CITATIONS": ('badge-cite', '✅ Cited answer'),
        "PARTIAL_ANSWER": ('badge-warn', '⚠️ Partial answer'),
    }.get(resp.state, ('badge-block', '🚫 Blocked'))
    badges = [f'<span class="badge {badge[0]}">{badge[1]}</span>',
              f'<span class="badge badge-info">Grounding {score:.0%}</span>',
              f'<span class="badge badge-info">Flagged {n_hal}/{total}</span>',
              f'<span class="badge badge-info">{latency:.1f}s</span>']
    if fallback:
        badges.append('<span class="badge badge-info">Retrieved passage, not generated</span>')
    st.markdown("".join(badges), unsafe_allow_html=True)

    if resp.state == "BLOCKED":
        st.error("I could not find enough support in the documents to answer this reliably. "
                 "Here is the closest evidence instead.")
        for i, chunk in enumerate((resp.retrieved_chunks or [])[:3]):
            st.markdown(
                f'<div class="evidence"><div class="evidence-head">Evidence {i + 1} · '
                f'{html.escape(str(chunk.get("source_file", "")))}, p.{html.escape(str(chunk.get("page_num", "")))} '
                f'· similarity {float(chunk.get("similarity_score", 0.0)):.3f}</div>'
                f'{_plain_text(chunk.get("text", ""))}</div>',
                unsafe_allow_html=True,
            )
    else:
        if resp.warning_message:
            st.caption(resp.warning_message)
        parts = []
        for v in res.verdicts:
            sentence = html.escape(v.sentence or "")
            if fallback or v.classification == "WEAKLY_SUPPORTED":
                parts.append(f'<span class="sent-weak">{sentence}</span>')
            elif v.classification == "SUPPORTED":
                cite = f'<span class="cite">[{html.escape(str(v.source_file))}, p.{html.escape(str(v.source_page))}]</span>'
                parts.append(f'<span class="sent-supported">{sentence}</span>{cite}')
            else:
                parts.append(f'<span class="sent-hallucinated">{sentence}</span>')
        st.markdown(f'<div class="answer">{" ".join(parts)}</div>', unsafe_allow_html=True)
        if resp.cited_sources:
            st.caption("Sources: " + ", ".join(resp.cited_sources))

    if st.toggle("Show explainability details", key=f"details_{idx}"):
        render_explainability_panel(res, idx)


# ─────────────────────────────────────────────────────────────────────────────
# Main chat
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header">
  <div class="title">🛡️ TrustRAG</div>
  <div class="subtitle">Offline RAG with explainable, multi-signal hallucination detection</div>
</div>
""", unsafe_allow_html=True)

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("result") is not None:
            render_answer(msg["result"], idx)
        else:
            st.markdown(msg["content"])

if len(st.session_state.messages) <= 1:
    st.markdown("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for col, (label, question) in zip(cols, EXAMPLES):
        if col.button(f"{label}: {question}", key=f"example_{question}", width="stretch"):
            st.session_state.pending_prompt = question
            st.rerun()

prompt = st.chat_input("Ask a question about the documents...")
if st.session_state.pending_prompt and not prompt:
    prompt = st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "result": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    pipe.hall_engine.w_nli, pipe.hall_engine.w_bert, pipe.hall_engine.w_cos = w_nli, w_bert, w_cos
    pipe.hall_engine.high_thr = pipe.controller.high_thr = high_threshold
    pipe.hall_engine.medium_thr = pipe.controller.medium_thr = medium_threshold

    with st.chat_message("assistant"):
        with st.status("Working on it...", expanded=True) as status:
            start = time.time()
            result = pipe.query(prompt, progress=lambda label: status.write(f"• {label}"))
            elapsed = time.time() - start
            status.update(label=f"Done in {elapsed:.1f}s", state="complete", expanded=False)

    if result.error:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Error: {result.error}", "result": None})
    else:
        result.summary["latency"] = elapsed
        st.session_state.messages.append({"role": "assistant", "content": "", "result": result})
    save_history()
    st.rerun()
