# Insti-Assist — Evaluation Report

- Index vectors: **247**
- Retrieval (n=27): **Hit@6 = 100.0%**, MRR = **0.921**

## Hit@k sweep

| k | Hit@k | MRR |
|---|-------|-----|
| 1 | 88.9% | 0.889 |
| 2 | 92.6% | 0.907 |
| 4 | 92.6% | 0.907 |
| 6 | 100.0% | 0.921 |
| 8 | 100.0% | 0.921 |

## MIN_SCORE ablation (score-gate only)

| MIN_SCORE | OOS refusal acc | False-refusal rate |
|-----------|-----------------|--------------------|
| 0.30 | 0.0% | 0.0% |
| 0.40 | 0.0% | 0.0% |
| 0.45 | 0.0% | 0.0% |
| 0.50 | 50.0% | 0.0% |
| 0.55 | 62.5% | 0.0% |
| 0.60 | 75.0% | 0.0% |
| 0.65 | 75.0% | 7.4% |
| 0.70 | 100.0% | 11.1% |

## Answer quality (end-to-end)

- Out-of-scope refusal accuracy: **100.0%**
- False-refusal rate: **18.5%**
- Faithfulness: **96.3%**