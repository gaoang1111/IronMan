python -m src.adaptive_segment.build_adaptive_chunks \
  --model_path /default-vepfs/public/user/ga/Iron/models/Llama-3.1-8B \
  --input_jsonl /default-vepfs/public/user/ga/Iron/IronMan/data/test_collator.jsonl \
  --output_jsonl /default-vepfs/public/user/ga/Iron/IronMan/data/test_collator_adaptive.jsonl \
  --device cuda \
  --dtype bf16 \
  --min_base_len 4 \
  --max_base_len 64 \
  --target_min_len 8 \
  --target_max_len 20 \
  --merge_upper_bound 24 \
  --event_boundary_threshold 5.0 \
  --split_max_len 24 \
  --split_min_len 8


# python -m src.adaptive_segment.build_adaptive_chunks \
#   --model_path /default-vepfs/public/user/ga/Iron/models/Llama-3.1-8B \
#   --input_jsonl /default-vepfs/public/user/ga/Iron/33/data/eval.jsonl \
#   --output_jsonl /default-vepfs/public/user/ga/Iron/IronMan/data/eval_adaptive.jsonl \
#   --device cuda \
#   --dtype bf16 \
#   --min_base_len 4 \
#   --max_base_len 64 \
#   --target_min_len 8 \
#   --target_max_len 20 \
#   --merge_upper_bound 24 \
#   --event_boundary_threshold 5.0 \
#   --split_max_len 24 \
#   --split_min_len 8