"""Qwen3Guard-Gen-8B 逐条审核 + label token logits 采集。

思路沿用 guard_reasoner.py：
    读 json list -> 逐条 apply_chat_template -> greedy generate(带 output_scores)
    -> 正则定位生成文本里的 Safety label -> 映射回 token 区间
    -> 采集 label 每个 token 的 top-2 分布 -> 写回原 item -> 整体 dump

与参考脚本的三处差异：
    1. 输入模板交给 Qwen3Guard 自带的 chat_template（不手写 INSTUCT，对齐 llama_guard.py）；
    2. label 是三档 Safe / Unsafe / Controversial + Categories 字段，不是二值；
    3. char->token 对齐改用 offset_mapping + 一致性校验，并修掉 guard_reasoner.py
       里 scores 下标偏移一位（range(label_token_start - 1, ...)）的问题。

首次运行前请确认：脚本会把第一条的渲染文本打印出来，务必肉眼核对模板是否符合模型卡。
"""

import json
import os
import re
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# === 配置路径 ===
MODEL_PATH = "/data/inspur/wangjunyi/qwen3guard-gen-8b"
INPUT_DIR = "/data/inspur/wangjunyi/toxic_test/sft_guardreasoner/test_data/test_data1"
FILE_NAME = "wildguardtest.json"
OUTPUT_DIR = "/data/inspur/wangjunyi/toxic_test/qwen3guard_logit/result/result1"

# === 运行参数 ===
MODE = "prompt_response"   # "prompt_response": 审核 (prompt, response) 对话；"prompt_only": 只审核 prompt
USE_THINK = False          # 开启思考模式后 scores 会多出推理段的步数，正则定位也更脆，默认关闭
MAX_NEW_TOKENS = 512
DEVICE = "cuda:0"
STRICT_CHECK = False       # True: top-1 与实际生成 token 不一致时直接抛错（首跑建议改成 True）
PRINT_FIRST_PROMPT = True  # 打印第一条的渲染文本，用来核对 chat_template

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载模型 ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto")
model.eval()
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
model.to(device)

# === Qwen3Guard 输出解析 ===
# 模型输出形如:
#   Safety: Unsafe
#   Categories: Violent
# 字段名以实际打印结果为准，如有出入只改这两条正则即可。
LABEL_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
CATEGORY_RE = re.compile(r"Categories:\s*([^\n]+)", re.IGNORECASE)


def render_prompt(messages):
    """走模型自带 chat_template；enable_thinking 只在模板认这个参数时才传。"""
    kwargs = {}
    if "enable_thinking" in (tokenizer.chat_template or ""):
        kwargs["enable_thinking"] = USE_THINK
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def char_span_to_token_span(gen_ids, gen_text, span_start, span_end):
    """把生成文本的字符区间 [span_start, span_end) 映射成 gen_ids 的下标区间。

    guard_reasoner.py 用的是 "prefix/full/suffix 各自重新分词再拼长度"，
    BPE 在边界处会合并导致下标整体偏移且不报错。这里先用 offset_mapping
    对齐真实 gen_ids，不一致时再用逐前缀 decode 兜底。
    返回 (start, end)，未命中返回 (None, None)。
    """
    # 方案 A：整段重新编码 + offset_mapping（要求重新编码的 token 与真实生成一致）
    enc = tokenizer(gen_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if ids[: len(gen_ids)] == gen_ids:
        idx = [k for k, (a, b) in enumerate(offsets) if b > span_start and a < span_end]
        if idx:
            return idx[0], idx[-1] + 1

    # 方案 B：逐个前缀 decode，字符长度单调递增，用它反查下标
    lens = [
        len(tokenizer.decode(gen_ids[: k + 1], skip_special_tokens=True))
        for k in range(len(gen_ids))
    ]
    start = next((k for k, n in enumerate(lens) if n > span_start), None)
    end = next((k for k, n in enumerate(lens) if n >= span_end), None)
    if start is None or end is None:
        return None, None
    return start, end + 1


# === 加载 JSON（list） ===
input_path = os.path.join(INPUT_DIR, FILE_NAME)
output_path = os.path.join(OUTPUT_DIR, FILE_NAME)
with open(input_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)  # 现在是一个列表

# === 遍历列表中的每个条目 ===
label_counter = Counter()
unmatched_count = 0
mismatch_count = 0

for i, item in enumerate(tqdm(data_list, desc="Processing")):
    prompt = (item.get("prompt") or "").strip()
    if prompt == "":
        continue
    response = (item.get("response") or "").strip()
    if response == "" or response is None:
        response = "None"

    messages = [{"role": "user", "content": prompt}]
    if MODE == "prompt_response":
        messages.append({"role": "assistant", "content": response})

    text = render_prompt(messages)
    if PRINT_FIRST_PROMPT and i == 0:
        print(f"\n===== 渲染后的输入（第 1 条）=====\n{text}\n=====\n")

    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy，保证可复现
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # 显式指定
        )

    gen_ids = gen.sequences[0][enc.input_ids.shape[1]:]  # 只取新生成部分
    scores = gen.scores  # scores[k] = 生成 gen_ids[k] 那一步的 logits
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    item["generated_text"] = gen_text
    item["safety_label_generated"] = None
    item["categories_generated"] = None

    match = LABEL_RE.search(gen_text)
    if not match:
        unmatched_count += 1
        print(f"[warn] 第 {i} 条未匹配到 Safety label，generated_text={gen_text!r}")
        continue

    label = match.group(1)
    item["safety_label_generated"] = label
    label_counter[label] += 1

    cat_match = CATEGORY_RE.search(gen_text)
    if cat_match:
        item["categories_generated"] = cat_match.group(1).strip()

    # === 定位 label 的 token 区间 ===
    label_token_start, label_token_end = char_span_to_token_span(
        gen_ids, gen_text, match.start(1), match.end(1)
    )
    if label_token_start is None:
        item["safety_label_token_span"] = None
        print(f"[warn] 第 {i} 条 label {label!r} 定位失败，退回文本结果")
        continue

    label_token_end = min(label_token_end, len(scores))
    item["safety_label_token_span"] = [label_token_start, label_token_end]

    # === 采集 label 每个 token 的 top-2 分布 ===
    for k in range(label_token_start, label_token_end):
        logits = scores[k].squeeze(0).float()  # [vocab]
        topk_logits, topk_indices = torch.topk(logits, 2)
        topk_probs = torch.softmax(topk_logits, dim=-1)  # top-2 之间重新归一化
        full_probs = torch.softmax(logits, dim=-1)       # 全词表 softmax

        # 自检：greedy 下实际生成的 token 必然是 argmax，不等说明下标错了
        if topk_indices[0].item() != gen_ids[k].item():
            mismatch_count += 1
            msg = (
                f"第 {i} 条 token {k}: top-1={tokenizer.decode([topk_indices[0].item()])!r} "
                f"!= 实际生成 {tokenizer.decode([gen_ids[k].item()])!r}，下标可能偏移"
            )
            if STRICT_CHECK:
                raise AssertionError(msg)
            print(f"[warn] {msg}")

        n = k - label_token_start + 1  # 与 guard_reasoner.py 一致，从 1 开始编号
        item[f"safety_label_token_top2_tokens_{n}"] = [
            tokenizer.decode([idx]) for idx in topk_indices.tolist()
        ]
        item[f"safety_label_token_top2_probs_{n}"] = [
            round(p, 6) for p in topk_probs.tolist()
        ]
        item[f"safety_label_token_full_probs_{n}"] = [
            round(full_probs[idx].item(), 6) for idx in topk_indices.tolist()
        ]

# === 保存整个列表到新 JSON 文件 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

# === 复现信息 ===
with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "model_path": MODEL_PATH,
            "input_path": input_path,
            "mode": MODE,
            "use_think": USE_THINK,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "device": str(device),
            "strict_check": STRICT_CHECK,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"\nlabel 分布: {dict(label_counter)}")
print(f"未匹配 label: {unmatched_count} / {len(data_list)}，argmax 不一致: {mismatch_count}")
print(f"结果已写入 {output_path}")
