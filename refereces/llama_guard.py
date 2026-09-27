import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_file_as_string(experiment_path):
    filename = f"{experiment_path}/need_to_add.txt"
    if not os.path.isfile(filename):
        return ""
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


# === 配置路径 ===
# experiment_path = "/data/inspur/wangjunyi/toxic_test/test_change_toxic/experiment6"
# input_dir = f"{experiment_path}/harmful_sentences_data3/"     # 你的json输入文件夹路径
# NEED_TO_ADD = NEED_TO_ADD = load_file_as_string(experiment_path)
output_dir = "/data/inspur/wangjunyi/toxic_test/llama_guard_logit/result/result3"   # 保存新json文件的目标文件夹路径
os.makedirs(output_dir, exist_ok=True)


# === 加载模型 ===
model_path = "/data/inspur/wangjunyi/toxic_test/my_toxic_test/model/models/fightinghc2024/Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 遍历文件夹中的 JSON 文件 ===
# for file_name in tqdm(os.listdir(input_dir), desc="Processing"):
#     if not file_name.endswith(".json"):
#         continue
if True:
    file_name = "wildguardtest.json"
    input_dir = "/data/inspur/wangjunyi/toxic_test/sft_guardreasoner/test_data/test_data1"
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    # === 加载 JSON（list） ===
    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f) # 现在是一个列表

    # === 遍历列表中的每个条目 ===
    for item in tqdm(data_list, desc="Processing"):
        prompt = item.get("prompt", "").strip()
        if prompt == "":
            continue
        response = item.get("response", "").strip()
        if response == "" or response == None:
            response = "None"
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)

        # === Step 0: 生成第一个 token ===
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            next_token_id = torch.argmax(last_logits).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id.to(device)], dim=1)

        # === Step 1: top-2 输出 ===
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            topk_logits, topk_indices = torch.topk(last_logits, 2)
            topk_probs = F.softmax(topk_logits, dim=-1)
            topk_tokens = [tokenizer.decode([idx]) for idx in topk_indices]
            topk_probs_list = [round(prob.item(), 6) for prob in topk_probs]

        # === 写回该元素中 ===
        item["step1_top_tokens"] = topk_tokens
        item["step1_top_probs"] = topk_probs_list

    # === 保存整个列表到新 JSON 文件 ===
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    # break
