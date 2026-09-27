"""选择本次训练用哪几张 GPU。

直接运行:
    python3 utils/select_GPU.py

作为模块用:
    from utils.select_GPU import select_gpus
    gpus = select_gpus()                      # 显示 nvidia-smi 并交互询问
    gpus = select_gpus(ask=False, count=2)    # 不问，直接挑最空闲的 2 张
    gpus = select_gpus(indices=[1, 2])        # 不问，就用 1、2 号

返回显卡编号列表（如 [1, 2]），可直接喂给 CUDA_VISIBLE_DEVICES。

本项目约定：所有要用显卡的任务都从这个函数拿编号，不要在别处硬编码卡号。

保护规则（没有例外，显式指定也不例外）：
    选中的卡必须都空闲，且数量不能超过空闲卡数。卡被别人占用时一律拦住，
    不会悄悄把任务塞到一张正在跑别的活的卡上。

挑卡规则：按空闲显存从多到少挑；显存相同时优先利用率低的。
"""

import subprocess
import sys

QUERY_FIELDS = "index,name,memory.used,memory.total,utilization.gpu"
IDLE_MIB = 1024  # 已用显存低于这个值就当成没人用，和表格里"空闲"的口径一致


def _smi(args):
    """跑一条 nvidia-smi。没装这个命令 / 没驱动 / 超时都返回 None。"""
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True,
                              text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _to_int(value):
    try:
        return int(value)
    except ValueError:
        return None


def show_smi():
    """原样打印 nvidia-smi 的结果，让用户看到全貌。成功返回 True。"""
    out = _smi([])
    if out is None:
        print("没找到 nvidia-smi，无法枚举显卡", file=sys.stderr)
        return False
    print((out.stdout or out.stderr).rstrip())
    print()
    return out.returncode == 0


def parse_gpus(text):
    """把 nvidia-smi 的 csv 输出解析成每张卡一个 dict。

    字段按「头一个 + 末三个」定位，中间的都算型号，这样型号里带逗号也不会串位。
    读成 [N/A] 的字段留 None，不猜成 0 —— 否则这张卡会被误判成最空闲的。
    """
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        index = _to_int(parts[0])
        if index is None:
            continue
        gpus.append({
            "index": index,
            "name": ",".join(parts[1:-3]).strip(),
            "mem_used": _to_int(parts[-3]),
            "mem_total": _to_int(parts[-2]),
            "util": _to_int(parts[-1]),
        })
    return gpus


def query_gpus():
    """枚举本机显卡，拿不到就返回空列表。"""
    out = _smi([f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"])
    if out is None or out.returncode != 0:
        return []
    return parse_gpus(out.stdout)


def free_mib(gpu):
    """空闲显存；读不出来返回 None。"""
    if gpu["mem_used"] is None or gpu["mem_total"] is None:
        return None
    return gpu["mem_total"] - gpu["mem_used"]


def is_idle(gpu):
    """判断一张卡是否空闲。显存读不出来的不算 —— 没法证明它空着。"""
    return gpu["mem_used"] is not None and gpu["mem_used"] < IDLE_MIB


def print_table(gpus):
    for g in gpus:
        free = free_mib(g)
        if free is None:
            mem, status = "显存未知", "未知"
        else:
            mem = f"显存 {g['mem_used']}/{g['mem_total']} MiB (空闲 {free} MiB)"
            status = "空闲" if is_idle(g) else "占用中"
        util = f"{g['util']}%" if g["util"] is not None else "未知"
        print(f"  [{g['index']}] {g['name']}   {mem}   利用率 {util}   {status}")


def pick_gpus(candidates, count):
    """从候选卡里挑 count 张最空闲的，按编号升序返回。

    显存读不出来的排最后：宁可挑明确空闲的，也不赌一张读不出数据的卡。
    """
    def rank(g):
        free = free_mib(g)
        return (free is None, -(free if free is not None else 0),
                g["util"] if g["util"] is not None else 0)

    return sorted(g["index"] for g in sorted(candidates, key=rank)[:count])


def reject_reason(chosen, gpus, idle):
    """chosen 有什么问题就返回理由字符串，没问题返回 None。"""
    by_index = {g["index"]: g for g in gpus}
    missing = [i for i in chosen if i not in by_index]
    if missing:
        return f"本机没有这些显卡编号 {missing}"
    busy = [i for i in chosen if i not in idle]
    if busy:
        detail = ", ".join(f"[{i}] 已用 {by_index[i]['mem_used']} MiB" for i in busy)
        return f"这些卡已被占用: {detail}"
    if len(chosen) > len(idle):
        return f"要 {len(chosen)} 张，但空闲卡只有 {len(idle)} 张"
    return None


def ask_count(n_idle, n_total):
    """交互询问用几张卡。返回张数；用户放弃（非终端 / Ctrl-C）返回 None。"""
    if n_idle == 0:
        print("当前没有空闲显卡，无法启动", file=sys.stderr)
        return None
    while True:
        try:
            raw = input(f"\n本次训练使用几张卡？（空闲 {n_idle} 张 / 共 {n_total} 张，"
                        f"直接回车=1）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nstdin 不是终端，没法交互选择；"
                  "改用 select_gpus(indices=[...]) 或 run_parallel.py --gpus 指定",
                  file=sys.stderr)
            return None
        if raw == "":
            count = 1
        elif raw.isdigit():
            count = int(raw)
        else:
            print(f"输入 {raw!r} 无效，请输入 1 ~ {n_idle} 的整数")
            continue
        if not 1 <= count <= n_idle:
            print(f"空闲卡只有 {n_idle} 张，用不了 {count} 张，请重新输入")
            continue
        return count


def select_gpus(ask=True, count=None, indices=None):
    """显示 nvidia-smi，选定显卡，返回编号列表。选不出来时返回空列表。

    ask=True 走交互，张数超出空闲数会重新问；
    ask=False 时按 count 张挑（不给就挑全部空闲卡）；
    indices 给定时直接用这些编号，跳过询问。
    后两种走非交互，被拦住时只打错误信息并返回 []，不会挂住等输入。
    """
    if not show_smi():
        return []
    gpus = query_gpus()
    if not gpus:
        print("nvidia-smi 没能枚举出任何显卡", file=sys.stderr)
        return []

    print_table(gpus)
    idle = {g["index"]: g for g in gpus if is_idle(g)}
    n_idle = len(idle)

    if indices is not None:
        chosen = sorted(set(indices))
    elif ask:
        count = ask_count(n_idle, len(gpus))
        if count is None:
            return []
        chosen = pick_gpus(list(idle.values()), count)
    else:
        if count is None:
            count = n_idle
        if n_idle == 0:
            print("当前没有空闲显卡，无法启动", file=sys.stderr)
            return []
        if not 1 <= count <= n_idle:
            print(f"要 {count} 张，但空闲卡只有 {n_idle} 张", file=sys.stderr)
            return []
        chosen = pick_gpus(list(idle.values()), count)

    reason = reject_reason(chosen, gpus, idle)
    if reason:
        print(f"选卡失败：{reason}", file=sys.stderr)
        return []

    by_index = {g["index"]: g for g in gpus}
    print(f"\n选用显卡: {chosen}")
    for i in chosen:
        g = by_index[i]
        free = free_mib(g)
        detail = f"空闲 {free} MiB" if free is not None else "显存未知"
        util = f"{g['util']}%" if g["util"] is not None else "未知"
        print(f"  [{i}] {g['name']}  {detail}  利用率 {util}")
    print(f"\n  export CUDA_VISIBLE_DEVICES={','.join(str(i) for i in chosen)}")
    return chosen


if __name__ == "__main__":
    if not select_gpus():
        sys.exit(1)
