BalDRO FQ Audit Sbatch
======================

These scripts are meant to isolate why TOFU `forget_quality` is unexpectedly low
in another project by running the same model/checkpoint through BalDRO's
Open-Unlearning-derived evaluator.

Recommended order on the HPC:

1. Submit `baldro_train_eval_npo_llama2_splits.sh`.
   - This is the main audit path: it runs NPO unlearning on BalDRO, then
     evaluates through BalDRO's evaluator.
   - It covers `forget01`, `forget05`, and `forget10`.
   - It uses the shared cache path `/home/zkzhang/unlearning/HF_CACHE` by default.

2. If reference logs are suspicious, submit `baldro_generate_retain_refs.sh`.
   - It regenerates retain-model eval logs into `results/baldro_fq_audit/retain_refs`.
   - Update or download the retain model paths before enabling all array items.

3. Optional: submit `baldro_eval_existing_models.sh`.
   - This evaluates already-existing full/unlearned checkpoints without training.
   - Use it only as a direct evaluator sanity check.

4. Optional: submit `baldro_train_eval_drnpo_llama2_forget01_grid.sh`.
   - This checks BalDRO-DV on `forget01` across the paper/repo beta-DV grid.

5. Submit `baldro_summarize_tofu.sh`.
   - This collects all `TOFU_SUMMARY.json` files into one CSV.

Common overrides:

```bash
sbatch sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
sbatch --array=0-0 sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
sbatch --export=ALL,EVAL_BATCH_SIZE=4 sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
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
- The default Hugging Face cache path is `/home/zkzhang/unlearning/HF_CACHE`.
