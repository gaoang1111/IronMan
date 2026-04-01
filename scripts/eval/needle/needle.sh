# python scripts/experiment/run_needle_scale.py \
#   --resume_path /default-vepfs/public/user/ga/Iron/33/checkpoints/phase-full-deepkv-group-gate/phase-full_step_500 \
#   --data_jsonl data/wikitext_long10.jsonl \
#   --out_dir data/needle_scale \
#   --num_positions 20 \
#   --tag "base"


python scripts/experiment/run_needle_scale.py \
  --resume_path /default-vepfs/public/user/ga/Iron/checkpoints/phase-full-dynamicq/phase-full_step_500 \
  --data_jsonl data/wikitext_long10.jsonl \
  --out_dir data/needle_scale \
  --num_positions 20 \
  --tag "q0"
  # --tag "dynamicq-1"
