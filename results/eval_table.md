# Baseline vs SFT vs DPO (held-out test split)

Base model: `Qwen/Qwen2.5-0.5B-Instruct` (tier `cpu_apple`, device `cpu`), 20 test tasks, 10000 bootstrap samples.

| variant | acc@1 | acc@1 95% CI | final acc | final 95% CI | turns-to-success | guardrail-viol rate | tok/task | ms/task | win/tie/loss vs baseline | win rate |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.250 | [0.100, 0.450] | 0.250 | [0.100, 0.450] | 1.00 | 0.100 | 1030 | 4 | - | - |
| sft | 0.250 | [0.100, 0.450] | 0.250 | [0.100, 0.450] | 1.00 | 0.200 | 999 | 7 | 1/18/1 | 0.050 |
| dpo | 0.200 | [0.050, 0.400] | 0.250 | [0.050, 0.450] | 1.20 | 0.157 | 1012 | 4 | 2/16/2 | 0.100 |

## Per-failure-mode delta vs baseline

| variant | correct | syntax_error | wrong_table | missing_join | bad_aggregation | wrong_filter | empty_result | timeout | guardrail_block |
|---|---|---|---|---|---|---|---|---|---|
| sft | 0 | 0 | -1 | 0 | 0 | 2 | -1 | 0 | 0 |
| dpo | 0 | 0 | 0 | 0 | 0 | 0 | -1 | 0 | 1 |
