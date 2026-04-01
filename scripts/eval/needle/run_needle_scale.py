from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def _parse_layers(spec: str) -> list[int]:
    parts = [p.strip() for p in str(spec).split(",")]
    return [int(p) for p in parts if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume_path", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, default="data/wikitext_long10.jsonl")
    ap.add_argument("--out_dir", type=str, default="data/needle_scale")
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--buffer_num", type=int, default=1)
    ap.add_argument("--q_num", type=int, default=None)
    ap.add_argument("--target_layers", type=str, default="15,23,31")
    ap.add_argument("--num_positions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_texts", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--tag", type=str, default="needle")
    args = ap.parse_args()
    print(args.tag)
    # exit()
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.experiment.needle import NeedleScaleConfig, run_needle_scale_from_jsonl
    from src.infra.infra import Mark42StreamingEngine
    from src.train_utils import load_model, load_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_args = SimpleNamespace(
        resume_path=str(args.resume_path),
        model_name="",
        load_weights_only=True,
        phase="phase1",
        javis_num_queries=(int(args.q_num) if args.q_num is not None else 2),
        javis_target_layers=_parse_layers(args.target_layers),
        chunk_size=int(args.chunk_size),
        buffer_num=int(args.buffer_num),
    )

    tokenizer, _ = load_tokenizer(dummy_args)
    model = load_model(dummy_args, tokenizer, device, is_resume=True)
    model = model.to(torch.bfloat16)

    engine = Mark42StreamingEngine(
        model,
        tokenizer,
        chunk_size=int(args.chunk_size),
        buffer_num=int(args.buffer_num),
        q_num=(int(args.q_num) if args.q_num is not None else None),
        target_layers=tuple(_parse_layers(args.target_layers)),
    )

    for i in range(7,11):
        cfg = NeedleScaleConfig(
        num_positions=int(args.num_positions),
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        repetition_penalty=float(args.repetition_penalty),
        tag=f"{args.tag}_offset_{i}",
        offset=i,
    )
        meta = run_needle_scale_from_jsonl(
            engine,
            tokenizer,
            data_jsonl_path=str(args.data_jsonl),
            out_dir=str(args.out_dir),
            config=cfg,
            max_texts=(int(args.max_texts) if args.max_texts is not None else None),
        )
        print(meta)

    # cfg = NeedleScaleConfig(
    #     num_positions=int(args.num_positions),
    #     seed=int(args.seed),
    #     max_new_tokens=int(args.max_new_tokens),
    #     temperature=float(args.temperature),
    #     repetition_penalty=float(args.repetition_penalty),
    #     tag=args.tag,
    # )


    # meta = run_needle_scale_from_jsonl(
    #     engine,
    #     tokenizer,
    #     data_jsonl_path=str(args.data_jsonl),
    #     out_dir=str(args.out_dir),
    #     config=cfg,
    #     max_texts=(int(args.max_texts) if args.max_texts is not None else None),
    # )
    # print(meta)


if __name__ == "__main__":
    main()

