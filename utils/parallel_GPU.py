"""把一个「逐条独立计算」的脚本分片并行跑在多张卡上，跑完合并成单个 json。

驱动侧（run_parallel.py 这种 CLI）:
    from utils.parallel_GPU import run, merge
    run(script, input_path, output_dir, gpus)   # 起进程 → 等 →（中断回收）→ 合并
    merge(input_path, output_dir)               # 只合并已有分片

工作侧（被跑的脚本）:
    from utils.parallel_GPU import Shards
    sh = Shards(all_items, input_path, output_dir)
    for i in tqdm(sh.todo, desc=sh.desc):
        ...
        sh.record(i, item)
    sh.close()

两侧之间唯一的耦合是下面这套约定，改这里就得同时改两边:
    run() 给每个子进程注入 CUDA_VISIBLE_DEVICES=单张卡、NUM_SHARDS、SHARD_ID、DEVICE=cuda:0。
    输入/输出路径不注入 env，由工作脚本自己声明，驱动侧得先问出同一份值再交给 run()/merge()
    （run_parallel.py 的做法是跑一次 `<脚本> --print-paths`），否则分片会落到 merge() 不看
    的地方、跑完不产出结果。
    子进程按 i % NUM_SHARDS 轮转分片，结果逐条 flush+fsync 追加到
    <输出目录>/<输入文件名>.shard<N>.jsonl（行格式 {"i": 原始下标, "item": {...}}），
    最后由 merge() 按原始下标拼回 <输入文件名>。

为什么是「一卡一进程」:
    每条样本互相独立，天然可数据并行，进程间零通信 —— 不需要 DDP/NCCL（那是给梯度同步
    用的），也不要用 device_map 那种模型并行（只会更慢）。代价是显存翻 N 倍。

分片为什么轮转切（i % N）而不是连续切块:
    样本耗时长短短差异大，连续切块会让某张卡摊到一堆长样本、早早跑完干等。

只有一张卡时也走分片:
    被 run() 托管（注入了 SHARD_ID）就走分片模式，分片数 = 卡数，1 张卡也照样落分片、
    由 merge 出正式 json —— 否则 worker 自己写了 json、merge 却按「一个分片都没有」报失败。
    没有 SHARD_ID 就是直接跑脚本，不分片，整体读入、跑完写整份 json。

中断续跑:
    分片结果逐条落盘，中断后重跑同一条命令即可，已完成的下标会被跳过。所以已完成下标要扫
    全部分片文件求并集：分片规则依赖卡数，续跑时卡数一变同一个下标就换了主人。

中断时会发生什么（Ctrl-C 或 kill 都走这条路）:
    1. 回收全部子进程 —— 走 utils/release_GPU.py，连后代一起杀，别把卡留着；
    2. 把已经跑完的部分合并出来（没跑完所以落在 *.partial.json）；
    3. 退出，重跑同一条命令从断点继续。
    为什么不只靠 Ctrl-C：SIGINT 会被 Python 推迟到当前 model.generate 返回才生效，
    子进程可能还要算完手上那条才死；这里补发 SIGTERM 让它们立刻退出。
"""
import glob
import json
import os
import signal
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from utils.release_GPU import release_processes


def _shard_paths(input_path, output_dir):
    return sorted(glob.glob(os.path.join(
        output_dir, f"{os.path.basename(input_path)}.shard*.jsonl")))


def _load_done_indices(shard_path):
    """读回一个分片 jsonl 中已完成的下标。"""
    done = set()
    if not os.path.isfile(shard_path):
        return done
    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["i"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # 中断时可能只写了一半的末行，丢掉即可
    return done


class Shards:
    """工作进程侧的分片账本：算出该跑哪些下标，逐条落盘。

    NUM_SHARDS / SHARD_ID 由 run() 注入；没有 SHARD_ID 说明是直接跑脚本，
    不分片，todo 就是全部下标，close() 时把 items 写成整份 json。
    """

    def __init__(self, items, input_path, output_dir):
        self.items = items
        self.input_path = input_path
        self.output_dir = output_dir

        if "SHARD_ID" not in os.environ:
            self.num_shards = 1
            self.shard_id = 0
            self.shard_path = None
            self.todo = list(range(len(items)))
            self.desc = "Processing"
            self._fh = None
            return

        self.num_shards = int(os.environ["NUM_SHARDS"])
        self.shard_id = int(os.environ["SHARD_ID"])
        self.shard_path = os.path.join(
            output_dir, f"{os.path.basename(input_path)}.shard{self.shard_id}.jsonl")

        # 已完成的下标要扫全部分片文件求并集，不能只读自己那份：分片规则 i % NUM_SHARDS
        # 依赖卡数，续跑时卡数一变，同一个下标就换了主人。只认自己的文件会把别人做过的
        # 重算一遍；卡数变少时更糟 —— 被砍掉那几张卡的成果全白干，因为它们的文件没人读。
        done = set()
        for path in _shard_paths(input_path, output_dir):
            done |= _load_done_indices(path)
        assigned = list(range(self.shard_id, len(items), self.num_shards))
        self.todo = [i for i in assigned if i not in done]
        self.desc = f"Processing shard {self.shard_id}/{self.num_shards}"
        self._fh = open(self.shard_path, "a", encoding="utf-8")
        # 同一个 NUM_SHARDS 下各分片的 assigned 互不相交，所以跨轮跳过是安全的、也不会抢同一条
        print(f"[shard {self.shard_id}/{self.num_shards}] 分到 {len(assigned)} 条，其中 "
              f"{len(assigned) - len(self.todo)} 条已完成，本次待跑 {len(self.todo)} 条 "
              f"-> {self.shard_path}")

    def record(self, item_idx, item):
        """把一条结果追加落盘。flush+fsync 保证进程被 kill 时这条也是完整的，
        开销相对单条的生成耗时可忽略；不这么做的话续跑会退回很久以前的进度。"""
        if self._fh is None:
            return
        self._fh.write(json.dumps({"i": item_idx, "item": item}, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        """收尾。分片模式只关文件句柄，结果由驱动侧的 merge() 拼；直接跑时落整份 json。"""
        if self._fh is None:
            output_path = os.path.join(self.output_dir, os.path.basename(self.input_path))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.items, f, ensure_ascii=False, indent=2)
            return
        self._fh.close()
        print(f"[shard {self.shard_id}/{self.num_shards}] 完成，本次写入 {len(self.todo)} 条 "
              f"-> {self.shard_path}")


def _install_sigterm_handler():
    """让 `kill <pid>` 也走 Ctrl-C 那条清理路径。

    SIGTERM 的默认动作是立刻终止，不给收尾机会，子进程就会变成孤儿继续占卡。
    """
    def handler(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:
        pass  # 不在主线程时设不了，忽略


def _launch(script, input_path, output_dir, gpus):
    """起 len(gpus) 个进程，各占一张卡、各跑一片数据。返回 [(shard_id, gpu, log, proc)]。

    起进程和等进程分开成两个函数，是为了中断时还能拿到进程句柄去回收。
    """
    with open(input_path, "r", encoding="utf-8") as f:
        total = len(json.load(f))

    base = os.path.basename(input_path)
    # 记下来这次用了哪些卡、多少条，供人查看
    with open(os.path.join(output_dir, f"{base}.manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"input_path": input_path, "num_shards": len(gpus),
                   "gpus": gpus, "total": total}, f, ensure_ascii=False, indent=2)

    procs = []
    for shard_id, gpu in enumerate(gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # 进程内只看得到这一张卡
        env["NUM_SHARDS"] = str(len(gpus))
        env["SHARD_ID"] = str(shard_id)
        env["DEVICE"] = "cuda:0"
        log_path = os.path.join(output_dir, f"{base}.shard{shard_id}.log")
        log = open(log_path, "w", encoding="utf-8")
        # -u 关掉输出缓冲，否则 tqdm 进度要等很久才刷进日志
        p = subprocess.Popen([sys.executable, "-u", script], env=env,
                             stdout=log, stderr=subprocess.STDOUT)
        print(f"[shard {shard_id}] GPU {gpu} 启动，pid {p.pid}，日志 {log_path}")
        procs.append((shard_id, gpu, log, p))
    return procs


def _collect(procs):
    """等所有进程退出，返回异常退出的 shard 编号。"""
    failed = []
    for shard_id, gpu, log, p in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            failed.append(shard_id)
            print(f"[shard {shard_id}] GPU {gpu} 异常退出，退出码 {rc}，见 {log.name}")
        else:
            print(f"[shard {shard_id}] GPU {gpu} 正常结束")
    return failed


def _shutdown(procs):
    """中断时回收子进程，别让它们变成孤儿继续占着卡。"""
    print(f"\n收到中断，回收 {len(procs)} 个子进程及其后代……")
    try:
        release_processes([p.pid for _, _, _, p in procs], ask=False)
    except KeyboardInterrupt:
        print("回收过程中又被中断，可能有进程还占着卡，用 "
              "`python3 utils/release_GPU.py --gpus <卡号>` 再收一次", file=sys.stderr)
    finally:
        for _, _, log, _ in procs:
            try:
                log.close()
            except OSError:
                pass


def merge(input_path, output_dir):
    """把各分片 jsonl 按原始下标拼回一个 list。

    全部下标记到齐才写 <输入文件名>；有缺失时改写 *.partial.json，
    免得半成品被下游当成完整结果。
    """
    if not os.path.isfile(input_path):
        print(f"[merge] 找不到输入文件 {input_path}，无法合并", file=sys.stderr)
        return False
    with open(input_path, "r", encoding="utf-8") as f:
        orig = json.load(f)
    total = len(orig)

    merged = {}
    for path in _shard_paths(input_path, output_dir):
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 中断时写了一半的末行
                merged[rec["i"]] = rec["item"]
                n += 1
        print(f"[merge] {os.path.basename(path)}: {n} 条")

    if not merged:
        print(f"[merge] 没有找到任何分片文件，什么都没写")
        return False

    # 缺失的下标（没跑到的）用原始 item 兜底，保证输出长度和输入一致
    out = [merged.get(i, orig[i]) for i in range(total)]
    missing = [i for i in range(total) if i not in merged]

    base = os.path.basename(input_path)
    if missing:
        out_path = os.path.join(output_dir, base.replace(".json", ".partial.json"))
        print(f"[merge] 还差 {len(missing)}/{total} 条（例如下标 {missing[:5]}），"
              f"说明还有分片没跑完，重跑同一条命令会从断点继续")
    else:
        out_path = os.path.join(output_dir, base)
        print(f"[merge] {total} 条全部到齐")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[merge] 已写入 {out_path}")
    return not missing


def run(script, input_path, output_dir, gpus):
    """整套流程：起 len(gpus) 个进程 → 等 →（中断则回收）→ 合并。返回条目是否全部到齐。"""
    os.makedirs(output_dir, exist_ok=True)
    _install_sigterm_handler()

    procs = _launch(script, input_path, output_dir, gpus)
    interrupted = False
    try:
        failed = _collect(procs)
    except KeyboardInterrupt:
        _shutdown(procs)
        interrupted = True
        failed = []
    if failed:
        print(f"\n分片 {failed} 没跑完。已完成的条目都在 jsonl 里，"
              f"修好后重跑同一条命令会从断点继续。")

    complete = merge(input_path, output_dir)
    if interrupted:
        print("被中断，已把跑完的部分合并出来。重跑同一条命令会从断点继续。")
    return complete
