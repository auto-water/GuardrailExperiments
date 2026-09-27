"""取 json 列表的前 N 条，存成新的 json。

用法:
    python3 utils/take_first.py 输入.json 输出.json          # 默认取前 10000 条
    python3 utils/take_first.py 输入.json 输出.json --n 100  # 指定条数
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description="取 json 列表前 N 条存成新 json")
    ap.add_argument("src", help="输入 json，顶层必须是列表")
    ap.add_argument("dst", help="输出 json")
    ap.add_argument("--n", type=int, default=10000, help="取前多少条，默认 10000")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{args.src} 顶层不是列表，实际是 {type(data).__name__}")

    sub = data[:args.n]
    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False, indent=2)
    print(f"已取前 {len(sub)}/{len(data)} 条 -> {args.dst}")


if __name__ == "__main__":
    main()
