# CareImpact AI benchmark report

Generated from saved results in `data/evaluation/results/`. Synthetic and real results are reported
separately; synthetic scores alone are not evidence of real-world policy accuracy.

## synthetic_test

| metric | careimpact-ft | diff-baseline | qwen2.5:1.5b |
|---|---|---|---|
| n | 240 | 240 | 20 |
| schema_valid_rate | 0.9958 | 1.0 | 1.0 |
| classification_accuracy | 0.9042 | 0.7083 | 0.05 |
| substantive_f1 | 0.9553 | 0.8857 | 0.2858 |
| finding_recall | 0.9109 | 0.9455 | 0.2353 |
| missed_important_change_rate | 0.0891 | 0.0545 | 0.7647 |
| unsupported_claim_rate | 0.5341 | 0.0 | 1.0 |
| evidence_exact_match_rate | 0.9 | 1.0 | — |
| section_coverage | 1.0 | 1.0 | 0.325 |
| no_change_accuracy | 0.9111 | 0.8 | 0.6667 |
| abstention_recall | 0.0 | 0.75 | — |
| false_abstention_rate | 0.0 | 0.0 | 0.0 |
| latency_mean_s | 39.124 | — | 19.753 |
