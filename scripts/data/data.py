import json
import os
from datasets import load_dataset
from pathlib import Path


dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

# 寻找单篇长度达标的词条
long_articles = []
current_article = []

for item in dataset:
    line = item["text"]
    # WikiText 的标题通常格式为 " = Title = "
    if line.strip().startswith("=") and not line.strip().startswith("= ="):
        if current_article:
            full_text = "".join(current_article)
            if len(full_text) > 30000: # 寻找超过 3万字符的单篇文章
                long_articles.append(full_text)
        current_article = [line]
    else:
        current_article.append(line)

# 取出第一篇真正的长文
if long_articles:
    final_text = long_articles[0]
    print(f"找到一篇长文，字符数: {len(final_text)}")
    file_path = Path("../data/wikitext_8k_eval.jsonl")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": final_text}, ensure_ascii=False) + "\n")
else:
    print("没找到足够长的单篇文章，建议放宽长度限制或增加数据源。")