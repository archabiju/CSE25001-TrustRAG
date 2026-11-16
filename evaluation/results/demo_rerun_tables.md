# Review 2 — measured results

37 questions / 109 labelled claims (39 hallucinated) · 213-chunk index · weights (0.45, 0.35, 0.2) · thresholds 0.75/0.5

## Overall (positive class = hallucinated)

| P | R | F1 | AUC-ROC | Acc |
|---|---|---|---|---|
| 0.7949 | 0.7949 | 0.7949 | 0.8952 | 0.8532 |

## Ablation

| Config | P | R | F1 @0.50 | AUC | best-F1 (thr) |
|---|---|---|---|---|---|
| NLI only | 0.8049 | 0.8462 | 0.825 | 0.9342 | 0.8471 (0.92) |
| BERTScore only | 0.0 | 0.0 | 0.0 | 0.7967 | 0.6667 (0.81) |
| Cosine only | 0.3929 | 0.5641 | 0.4632 | 0.5744 | 0.5417 (0.75) |
| NLI + BERTScore | 0.7895 | 0.7692 | 0.7792 | 0.9143 | 0.8539 (0.9) |
| TrustRAG (all three) | 0.7949 | 0.7949 | 0.7949 | 0.8952 | 0.8395 (0.69) |

## Evidence windows vs whole-chunk premises (all three signals)

| Premise | P | R | F1 @0.50 | AUC | best-F1 (thr) |
|---|---|---|---|---|---|
| whole 512-token chunk | 0.3936 | 0.9487 | 0.5564 | 0.6863 | 0.5913 (0.4) |
| 2-sentence windows (shipped) | 0.7949 | 0.7949 | 0.7949 | 0.8952 | 0.8395 (0.69) |

## Tri-state distribution (% of claims)

| Claim type | Supported | Weak | Hallucinated |
|---|---|---|---|
| Supported claims | 85.7 | 2.9 | 11.4 |
| Hallucinated claims | 10.3 | 10.3 | 79.5 |

## Retrieval (doc-level, k=5)

Recall@5 0.7973 · MRR 0.7905 · NDCG@5 0.7669 · Hit@5 0.8378