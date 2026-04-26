# Defense Audit - 2026-04-06

Audit rule applied: each answer is marked PASS or FAIL/INCOMPLETE with artifact evidence.

## Section 1 - Run Completeness

1. FAIL - Not all runs completed intended rounds; at least one run has empty rounds_data after failure. Evidence: [experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json)
2. FAIL - Round-count parity across all compared methods is not fully evidenced in one common artifact set. Evidence: [outputs/phase2_q1/stage6_baselines.json](outputs/phase2_q1/stage6_baselines.json), [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
3. FAIL/INCOMPLETE - No unified run log proving absence of NaN/OOM/timeout for all runs; only per-seed error arrays for some runs. Evidence: [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json), [experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json)
4. FAIL - Missing-round risk exists due failed runs with zero logged rounds. Evidence: [experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json)
5. FAIL - Seed completeness is inconsistent by experiment (for example 6 seeds instead of 10). Evidence: [experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json), [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json)
6. PASS (red-flag detected) - Missing/skipped seeds are present in prototype shards. Evidence: [experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json), [experiments/prototype_high_het_10seed_20260322_185816_6d9760a9/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_185816_6d9760a9/per_seed_results.json)
7. FAIL - Some seeds have invalid terminal outcomes (all-zero metrics with explicit error). Evidence: [experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json)
8. PASS (red-flag detected) - Unusually low communication exists (total_bytes_sent=0 for all entries in multiple files). Evidence: [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json), [experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json)
9. PASS (red-flag detected) - Logging stops with no convergence trajectory in failed run. Evidence: [experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json)

## Section 2 - Configuration Consistency

10. FAIL/INCOMPLETE - Identical architecture/init/split is only enforced for fedavg vs prototype in current blueprint code path, not fully evidenced for MOON/SCAFFOLD in the same artifact family. Evidence: [scripts/execute_q1_blueprint.py](scripts/execute_q1_blueprint.py), [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
11. FAIL/INCOMPLETE - Preprocessing consistency not logged as a cross-method runtime assertion artifact. Evidence: [outputs/q1_blueprint/execution_summary.json](outputs/q1_blueprint/execution_summary.json)
12. FAIL/INCOMPLETE - Augmentation pipeline parity not explicitly logged. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
13. PASS (configured) - Stage-6 baseline script configures batch_size=64 across methods. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py), [outputs/phase2_q1/stage6_baselines.json](outputs/phase2_q1/stage6_baselines.json)
14. PASS (configured) - Stage-6 sets local_epochs=3 for all methods. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
15. PASS (configured) - Stage-6 sets rounds=20 for all methods. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
16. FAIL/INCOMPLETE - Communication-budget equalization is not used nor documented as alternative control. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
17. PASS (configured) - Learning rate is aligned at 0.003 in stage-6 setup. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
18. PASS (configured) - Weight decay appears default-consistent (0) across Adam calls in stage-6 paths. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
19. PASS (configured) - Optimizer is Adam across stage-6 methods. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py), [flex_persona/training/optim_factory.py](flex_persona/training/optim_factory.py)

## Section 3 - Dataset Validity

20. FAIL/INCOMPLETE - No finalized artifact proving zero overlap for train/test and client partitions across all compared runs. Evidence: [outputs/q1_blueprint/phase_0.json](outputs/q1_blueprint/phase_0.json), [outputs/q1_blueprint/phase_1.json](outputs/q1_blueprint/phase_1.json)
21. FAIL/INCOMPLETE - Split reuse is asserted in code but not evidenced in the current stored phase-4 artifact (older run). Evidence: [scripts/execute_q1_blueprint.py](scripts/execute_q1_blueprint.py), [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
22. FAIL - Per-client class distribution by alpha is not logged for all required alpha runs in finalized artifacts. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
23. FAIL - Alpha 0.1 skew proof is not published with a dedicated distribution artifact/plot. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
24. FAIL - Heterogeneity quantification (entropy/KL/hist) not present in saved phase-4 output. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
25. PASS (for recorded phase-1 run) - No empty clients in phase-1 manifest (min_samples=128). Evidence: [outputs/q1_blueprint/phase_1.json](outputs/q1_blueprint/phase_1.json)
26. PASS (for recorded phase-1 run) - No client with missing class map in phase-1 manifest. Evidence: [outputs/q1_blueprint/phase_1.json](outputs/q1_blueprint/phase_1.json)

## Section 4 - Metric Validity

27. PASS (where detailed rounds exist) - Individual client accuracies are logged in global_metrics.client_accuracies. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
28. PASS (where detailed rounds exist) - Per-client local loss is logged. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
29. PASS (spot-check) - Mean aligns with recomputation from raw client accuracies. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
30. PASS (spot-check) - Worst-client matches minimum client accuracy. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
31. FAIL/INCOMPLETE - Std semantics cannot be fully validated for publication tables from existing artifacts. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json), [outputs/phase2_q1/stage6_baselines.json](outputs/phase2_q1/stage6_baselines.json)
32. PASS (spot-check) - Reported percentile spread matches computed p90-p10 in representative run. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
33. PASS (spot-check) - Mean lies within min/max client accuracies in representative run. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
34. PASS (spot-check) - No out-of-range accuracy observed in checked artifacts. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json), [experiments/comprehensive_10seed_results.json](experiments/comprehensive_10seed_results.json)
35. FAIL - Non-smooth and abrupt dynamics are present (collapse flags, oscillatory seed traces). Evidence: [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json), [experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json)

## Section 5 - Communication Accounting

36. PASS (implementation-level) - Model-size accounting uses params x 4 bytes in simulator. Evidence: [flex_persona/federated/simulator.py](flex_persona/federated/simulator.py)
37. PASS (implementation-level) - Upload and download are counted each round. Evidence: [flex_persona/federated/simulator.py](flex_persona/federated/simulator.py)
38. FAIL - Several per-seed logs report zero total communication. Evidence: [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json), [experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json)
39. FAIL/INCOMPLETE - Scaling with clients and rounds is not evidenced in a consolidated finalized communication benchmark artifact. Evidence: [outputs/reports/fedavg_curve_30_tuned_syncfix_report.json](outputs/reports/fedavg_curve_30_tuned_syncfix_report.json), [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
40. FAIL - Communication definitions differ by method path (state_dict bytes vs structured message serialization), preventing strict like-for-like accounting without normalization artifact. Evidence: [flex_persona/federated/simulator.py](flex_persona/federated/simulator.py)
41. FAIL - Some method result files omit server-to-client values (total_bytes_sent=0 only). Evidence: [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json)

## Section 6 - Convergence Validity

42. FAIL - No convergence plot files found in artifact directories. Evidence: [outputs](outputs), [outputs/reports](outputs/reports)
43. FAIL - Curves are not consistently smooth; multiple runs are flat/unstable. Evidence: [experiments/comprehensive_10seed_results.json](experiments/comprehensive_10seed_results.json), [outputs/reports/fedavg_curve_30_tuned_syncfix_report.json](outputs/reports/fedavg_curve_30_tuned_syncfix_report.json)
44. FAIL - Multi-seed trends are not consistently aligned across methods. Evidence: [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json), [experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_200753_cb4e8d4f/per_seed_results.json)
45. PASS (failure modes detected) - Divergence/collapse indicators are present via collapsed flags and hard errors. Evidence: [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json), [experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json)

## Section 7 - Scaling Diagnosis Completeness

46. FAIL - 10-client degradation experiment in q1 blueprint is not full-budget (phase-4 currently logs 1 round). Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
47. FAIL - Variables were swept in a grid, not isolated one-at-a-time diagnosis. Evidence: [outputs/q1_blueprint/phase_2.json](outputs/q1_blueprint/phase_2.json)
48. FAIL - Consistency across seeds is not established in current q1 artifacts. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
49. FAIL - No plots showing response vs local_epochs and learning_rate. Evidence: [outputs](outputs), [outputs/reports](outputs/reports)
50. FAIL/INCOMPLETE - Root-cause claim is not backed by finalized full-budget evidence package. Evidence: [outputs/q1_blueprint/phase_10.json](outputs/q1_blueprint/phase_10.json)

## Section 8 - Non-IID Experiment Completeness

51. FAIL - Each alpha has only 1 seed in saved phase-4 artifact (needs >=5). Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
52. FAIL - Full rounds not used in saved phase-4 artifact (1 round). Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
53. FAIL/INCOMPLETE - Same-split proof across methods is not present in saved phase-4 artifact. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
54. FAIL - Worst-client improvement is not consistently demonstrated across alphas in saved artifact. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
55. FAIL - Stability across seeds cannot be shown (single-seed per alpha artifact). Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
56. FAIL - Variance reduction not established in non-IID full protocol. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
57. PASS (red-flag detected) - Apparent improvement is currently tied to extremely limited seed evidence. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
58. PASS (no direct red-flag in saved phase-4 table) - No case where mean improved while worst degraded is visible in current phase-4 rows. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)

## Section 9 - Ablation Completeness

59. FAIL - Full ablation set (base, cluster-only, center-only, full) is not present in finalized q1 artifacts. Evidence: [outputs/q1_blueprint](outputs/q1_blueprint), [outputs/reports/ablation_smoke_recheck_20260320T105706Z.json](outputs/reports/ablation_smoke_recheck_20260320T105706Z.json)
60. FAIL - Current ablation evidence uses only 2 seeds in smoke artifact. Evidence: [outputs/reports/ablation_smoke_recheck_20260320T105706Z.json](outputs/reports/ablation_smoke_recheck_20260320T105706Z.json)
61. FAIL - Component contribution is not quantifiable from incomplete variant set. Evidence: [outputs/reports/ablation_smoke_recheck_20260320T105706Z.json](outputs/reports/ablation_smoke_recheck_20260320T105706Z.json)
62. FAIL/INCOMPLETE - Contradiction analysis not documented in finalized artifact. Evidence: [outputs/q1_blueprint](outputs/q1_blueprint)

## Section 10 - Statistical Validation

63. FAIL/INCOMPLETE - Paired testing logic exists in code, but no saved phase-6 output artifact in current run set. Evidence: [scripts/execute_q1_blueprint.py](scripts/execute_q1_blueprint.py), [outputs/q1_blueprint](outputs/q1_blueprint)
64. FAIL - No p-values reported in saved q1 outputs. Evidence: [outputs/q1_blueprint](outputs/q1_blueprint)
65. FAIL - No effect-size artifact reported in saved q1 outputs. Evidence: [outputs/q1_blueprint](outputs/q1_blueprint)
66. FAIL - Effective non-IID sample size in saved phase-4 is n=1 per alpha/method. Evidence: [outputs/q1_blueprint/phase_4.json](outputs/q1_blueprint/phase_4.json)
67. FAIL - Statistical significance cannot be established without phase-6 outputs and adequate n. Evidence: [outputs/q1_blueprint](outputs/q1_blueprint)

## Section 11 - Baseline Comparison Validity (MOON & SCAFFOLD)

68. FAIL/INCOMPLETE - Implementations exist but correctness is not validated by independent baseline verification artifact. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py), [outputs/phase2_q1/stage6_baselines.json](outputs/phase2_q1/stage6_baselines.json)
69. FAIL - No hyperparameter tuning artifact for MOON/SCAFFOLD (default-run evidence only). Evidence: [outputs/phase2_q1/stage6_baselines.json](outputs/phase2_q1/stage6_baselines.json)
70. PASS (configured) - Stage-6 baseline script uses matched seeds/rounds/local_epochs/alpha configuration. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
71. FAIL/INCOMPLETE - Equal communication budget is not reported for MOON/SCAFFOLD outputs. Evidence: [outputs/phase2_q1/stage6_baselines.json](outputs/phase2_q1/stage6_baselines.json)
72. PASS - Stage-6 averages all methods over the same seeds list. Evidence: [scripts/phase2_q1_validation.py](scripts/phase2_q1_validation.py)
73. PASS (red-flag detected) - SCAFFOLD performance is abnormally low, suggesting implementation/tuning issue. Evidence: [outputs/phase2_q1/stage6_baselines.json](outputs/phase2_q1/stage6_baselines.json)
74. PASS (red-flag detected) - Baseline trend likely contradicts common literature expectations for SCAFFOLD. Evidence: [outputs/phase2_q1/stage6_baselines.json](outputs/phase2_q1/stage6_baselines.json)

## Section 12 - Robustness

75. FAIL - Robustness across seeds, client counts, and alpha is not fully demonstrated in finalized outputs. Evidence: [outputs/q1_blueprint](outputs/q1_blueprint)
76. FAIL - Worst-case scenario reporting is not consolidated in final artifact package. Evidence: [outputs/q1_blueprint/phase_10.json](outputs/q1_blueprint/phase_10.json)

## Section 13 - Result Traceability

77. FAIL - Not every table number is traceable to run_id/seed/log; traceability table is empty. Evidence: [outputs/q1_blueprint/traceability_table.json](outputs/q1_blueprint/traceability_table.json)
78. FAIL - Mapping document exists as a file but currently contains no rows. Evidence: [outputs/q1_blueprint/traceability_table.json](outputs/q1_blueprint/traceability_table.json)

## Section 14 - Failure Mode Documentation

79. PASS (partial) - Failure cases are present in artifacts (errors and collapse flags). Evidence: [experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json](experiments/prototype_high_het_10seed_20260322_181730_dee26092/per_seed_results.json), [experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json](experiments/fedavg_high_het_10seed_20260322_200801_1ba6f569/per_seed_results.json)
80. FAIL/INCOMPLETE - Explanations are not systematically documented in a dedicated failure-analysis report. Evidence: [outputs/q1_blueprint](outputs/q1_blueprint)

## Section 15 - Final Claim Validation

81. PASS (template only) - Problem statement is explicitly defined. Evidence: [outputs/q1_blueprint/phase_10.json](outputs/q1_blueprint/phase_10.json)
82. FAIL - Conditions of outperformance are not finalized with completed evidence. Evidence: [outputs/q1_blueprint/phase_10.json](outputs/q1_blueprint/phase_10.json)
83. FAIL - Quantified gain is not finalized in claim artifact (TBD path remains). Evidence: [outputs/q1_blueprint/phase_10.json](outputs/q1_blueprint/phase_10.json)
84. FAIL - Communication/compute cost claim is not finalized in claim artifact. Evidence: [outputs/q1_blueprint/phase_10.json](outputs/q1_blueprint/phase_10.json)
85. PASS (mechanistic hypothesis present) - Mechanism is stated in claim template. Evidence: [outputs/q1_blueprint/phase_10.json](outputs/q1_blueprint/phase_10.json)
86. FAIL - Failure conditions are not finalized in claim artifact (limitation remains TBD). Evidence: [outputs/q1_blueprint/phase_10.json](outputs/q1_blueprint/phase_10.json)

## Final Pass Criteria Assessment

- No unanswered questions: PASS (all answered)
- No missing evidence: FAIL
- No inconsistent configs: FAIL
- All experiments full rounds/full seeds/identical conditions: FAIL

Verdict: NOT PUBLISHABLE under this defense audit.
