# results/ — measured artifacts

Every file here is produced by an actual run of the pipeline on the hardware
recorded in the model cards. Nothing is hand-written.

| file | produced by | what it contains |
|---|---|---|
| `dataset_stats.json` | `make label` | trace counts, SFT examples, DPO pairs, per-failure-mode rejected counts, repair-derived example count, split sizes, leakage-check status |
| `train_sft_log.json` | `make sft` | SFT config, model card, and the trainer log history (loss curve) |
| `train_dpo_log.json` | `make dpo` | DPO config, model card, reward margins (`rewards/chosen`, `rewards/rejected`, `rewards/margins`) and log history |
| `eval_report.json` | `make eval` | full baseline/SFT/DPO comparison: metrics, 10k-sample bootstrap 95% CIs, paired win-rate deltas, per-task results, per-failure-mode deltas |
| `eval_table.md` | `make eval` | the same comparison as a markdown table |
| `fig_accuracy_ci.png` | `make eval` | turn-1 and final execution accuracy with bootstrap CIs |
| `fig_failure_modes.png` | `make eval` | final failure-mode counts per variant |
| `fig_cost_delta.png` | `make eval` | token and latency cost deltas versus baseline |
| `loop_history.json` | `make loop` | one record per continuous-learning cycle: gate decision, reasons, gate details, drift, datasets, model cards |
| `loop/cycle*_sft_log.json`, `loop/cycle*_dpo_log.json` | `make loop` | per-cycle training logs |
| `drift.json` | drift stage | PSI/KL between training and live trace distributions per feature |
| `dashboard.html` | `make dashboard` | static, self-contained dashboard rendered from the registry and eval history |

The model registry itself lives at `artifacts/registry.json` (git-ignored); a
copy of its decisions is embedded in `loop_history.json` and `dashboard.html`.
