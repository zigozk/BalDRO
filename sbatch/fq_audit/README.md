BalDRO FQ Audit Sbatch
======================

These scripts are meant to isolate why TOFU `forget_quality` is unexpectedly low
in another project by running the same model/checkpoint through BalDRO's
Open-Unlearning-derived evaluator.

Recommended order on the HPC:

1. Submit `baldro_eval_existing_models.sh`.
   - This uses BalDRO's bundled retain reference logs under `saves/eval`.
   - It is the fastest check for whether BalDRO's evaluator reproduces the low FQ.

2. If reference logs are still suspicious, submit `baldro_generate_retain_refs.sh`.
   - It regenerates retain-model eval logs into `results/baldro_fq_audit/retain_refs`.
   - Update or download the retain model paths before enabling all array items.

3. If evaluator behavior looks sane, submit `baldro_train_eval_npo_llama2_splits.sh`.
   - This trains BalDRO's NPO baseline on `forget01`, `forget05`, and `forget10`.
   - It evaluates at every epoch through BalDRO's trainer.

4. Optional: submit `baldro_train_eval_drnpo_llama2_forget01_grid.sh`.
   - This checks BalDRO-DV on `forget01` across the paper/repo beta-DV grid.

5. Submit `baldro_summarize_tofu.sh`.
   - This collects all `TOFU_SUMMARY.json` files into one CSV.

Common overrides:

```bash
sbatch --export=ALL,ROOT=/home/zkzhang/unlearning/BalDRO,CONDA_ENV=baldro sbatch/fq_audit/baldro_eval_existing_models.sh
sbatch --export=ALL,EVAL_BATCH_SIZE=4 sbatch/fq_audit/baldro_eval_existing_models.sh
sbatch --export=ALL,MODEL_ROOT=/home/zkzhang/models sbatch/fq_audit/baldro_generate_retain_refs.sh
```

Notes:

- `src/train.py` only enters training when `trainer.args.report_to` is `wandb`
  or `mlflow`, so the training scripts set `WANDB_MODE=offline` and
  `trainer.args.report_to=wandb`.
- BalDRO already includes retain reference logs for several TOFU models in
  `saves/eval`. Regenerating them is optional, but useful for ruling out a
  reference-log mismatch.
- The default paths assume models are under `/home/zkzhang/models`.
