"""回收自己起的进程，把显卡腾出来。

常用法:
    python3 utils/release_GPU.py 1234 5678      # 按 pid 回收（连同它们的后代）
    python3 utils/release_GPU.py --gpus 1,2     # 不知道 pid 时按卡号反查自有进程
    python3 utils/release_GPU.py 1234 --yes     # 跳过确认（脚本里调用）

作为模块用:
    from utils.release_GPU import release_processes, pids_on_gpus
    release_processes([p.pid for p in procs])   # 传给 run_parallel 的子进程 pid

为什么不能只杀这个 pid 本身:
    进程被杀后，它的子进程会被 reparent 到 init(pid 1)，父子关系就断了 ——
    等你想再顺着树去找就找不回来了。所以这里先扫 /proc 把整棵树记下来，再统一发信号。
    另一个原因：直接 kill 父进程，子进程可能活着继续占卡。

只回收自己的进程:
    默认只动 uid 和当前进程一致的进程。别人的进程即使占着卡也不会碰，只会提示。
"""

import argparse
import os
import signal
import subprocess
import sys
import time

SIGTERM_WAIT = 10.0  # 发完 SIGTERM 等多久，还活着就升级成 SIGKILL


def _proc_stat(pid):
    """读 /proc/<pid>/stat，返回 (ppid, state)；进程不在了返回 None。"""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    # comm 字段带括号、里面可能有空格和右括号，从最后一个 ')' 之后切才稳
    rest = data[data.rindex(")") + 2:].split()
    if len(rest) < 2:
        return None
    return int(rest[1]), rest[0]  # (ppid, state)


def _pid_alive(pid):
    """进程是否还活着。僵尸不算 —— 它已经退出、显存早释放了，只是父进程还没收尸。"""
    stat = _proc_stat(pid)
    return stat is not None and stat[1] != "Z"


def _cmdline_of(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return "?"
    parts = [p for p in raw.split("\0") if p]
    return " ".join(parts) if parts else "?"


def _short_cmd(pid, limit=70):
    cmd = _cmdline_of(pid)
    return cmd if len(cmd) <= limit else cmd[: limit - 3] + "..."


def _uid_of(pid):
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def _ancestors_of(pid):
    """pid 的所有祖先，用来防止把自己或自己的父进程杀掉。"""
    out, seen = [], {pid}
    while pid and pid > 1:
        stat = _proc_stat(pid)
        if stat is None:
            break
        pid = stat[0]
        if pid <= 1 or pid in seen:
            break
        seen.add(pid)
        out.append(pid)
    return out


def descendants(pids):
    """把给的 pid 展开成「它们自己 + 所有后代」，按层序返回。

    必须一次性扫完再动手：父进程一死，子进程就 reparent 到 init，父子链就断了。
    """
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        child = int(entry)
        stat = _proc_stat(child)
        if stat is not None:
            children.setdefault(stat[0], []).append(child)

    ordered, seen = [], set()
    queue = list(pids)
    while queue:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        queue.extend(children.get(pid, []))
    return ordered


def compute_apps():
    """nvidia-smi 里的计算进程，返回 [{pid, gpu, mem_mib}]。拿不到返回 []。"""
    def smi(fields):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-{fields}",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return out.stdout if out.returncode == 0 else None

    uuid_out = smi("gpu=index,uuid")
    apps_out = smi("compute-apps=pid,gpu_uuid,used_memory")
    if uuid_out is None or apps_out is None:
        return []
    index_of_uuid = {}
    for line in uuid_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            index_of_uuid[parts[1]] = int(parts[0])
    apps = []
    for line in apps_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            apps.append({"pid": int(parts[0]),
                         "gpu": index_of_uuid.get(parts[1]),
                         "mem_mib": int(parts[2])})
        except ValueError:
            continue
    return apps


def pids_on_gpus(gpu_ids):
    """反查哪些自有进程正占着这些卡。别人的进程不返回。"""
    me = os.getuid()
    wanted, seen = set(gpu_ids), set()
    out = []
    for app in compute_apps():
        if app["gpu"] in wanted and app["pid"] not in seen and _uid_of(app["pid"]) == me:
            seen.add(app["pid"])
            out.append(app["pid"])
    return sorted(out)


def describe(pids, apps=None):
    """把一组 pid 打印成人看的一行行，附带它们在卡上占了多少显存。

    apps 预传进来可以少跑几次 nvidia-smi。
    """
    apps = compute_apps() if apps is None else apps
    me = os.getuid()
    for pid in pids:
        uid = _uid_of(pid)
        owner = "" if uid == me else f"  [uid={uid} 不是本用户!]"
        mem = [f"GPU {a['gpu']}: {a['mem_mib']} MiB" for a in apps if a["pid"] == pid]
        print(f"  {pid:>8}  {'  '.join(mem) if mem else '未占显存':<28} {_short_cmd(pid)}{owner}")


def release_processes(pids, ask=True, timeout=SIGTERM_WAIT):
    """回收这些 pid 及其全部后代。返回没杀掉的 pid 列表。

    先发 SIGTERM（Python 默认不拦它，进程会立刻退出，显存马上还给驱动），
    等 timeout 秒还活着的升级成 SIGKILL。别人的进程一律不碰。
    """
    pids = [int(p) for p in pids]
    protected = {os.getpid(), *_ancestors_of(os.getpid())}
    me = os.getuid()

    targets = [p for p in descendants(pids) if p not in protected]
    if len(targets) != len(pids):
        extra = len(targets) - len(pids)
        print(f"展开后代：给了 {len(pids)} 个 pid，实际要回收 {len(targets)} 个"
              + (f"（多出 {extra} 个后代）" if extra > 0 else ""))

    alive = [p for p in targets if _pid_alive(p)]
    if not alive:
        print("这些进程都已经不在了，无需回收")
        return []

    mine = [p for p in alive if _uid_of(p) == me]
    others = [p for p in alive if _uid_of(p) != me]
    if others:
        print(f"跳过 {len(others)} 个不属于本用户的进程：{others}", file=sys.stderr)
    if not mine:
        print("没有属于本用户的进程可回收", file=sys.stderr)
        return others

    print(f"\n将要回收 {len(mine)} 个进程：")
    describe(mine)

    if ask:
        try:
            if input(f"\n确认终止这 {len(mine)} 个进程？[y/N] ").strip().lower() not in ("y", "yes"):
                print("已取消")
                return alive
        except (EOFError, KeyboardInterrupt):
            print("\n不是终端，未确认，已取消", file=sys.stderr)
            return alive

    for pid in mine:  # 先 SIGTERM：让它们有机会正常退出
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"  {pid} 没权限终止", file=sys.stderr)

    deadline = time.time() + timeout
    while time.time() < deadline and any(_pid_alive(p) for p in mine):
        time.sleep(0.2)

    stubborn = [p for p in mine if _pid_alive(p)]
    if stubborn:
        print(f"\n{len(stubborn)} 个进程没响应 SIGTERM，升级为 SIGKILL：{stubborn}")
        for pid in stubborn:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1.0)

    survivors = [p for p in mine if _pid_alive(p)]
    freed = [p for p in mine if p not in survivors]
    print(f"\n已回收 {len(freed)}/{len(mine)} 个进程")
    if survivors:
        print(f"仍存活：{survivors}（可能卡在不可中断的系统调用里，需要 sudo 处理）",
              file=sys.stderr)
    # 进程没了不代表 nvidia-smi 立刻就不显示它，驱动释放有个短暂的滞后
    lingering = sorted({a["gpu"] for a in compute_apps()
                        if a["pid"] in set(freed) and a["gpu"] is not None})
    if lingering:
        print(f"注意：这些卡的显存还没立刻释放，等几秒再 nvidia-smi 看：{lingering}")
    return survivors + others


def main():
    ap = argparse.ArgumentParser(description="回收自己起的进程，把显卡腾出来")
    ap.add_argument("pids", nargs="*", type=int, help="要回收的进程 pid（连同其后代）")
    ap.add_argument("--gpus", help="按卡号反查自有进程，如 1,2")
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    args = ap.parse_args()

    pids = list(args.pids)
    if args.gpus:
        try:
            gpu_ids = [int(x) for x in args.gpus.split(",")]
        except ValueError:
            sys.exit(f"--gpus 格式不对: {args.gpus!r}，应形如 1,2")
        found = pids_on_gpus(gpu_ids)
        if not found:
            print(f"GPU {gpu_ids} 上没有本用户的计算进程")
            return
        pids.extend(found)

    if not pids:
        sys.exit("没给 pid，也没给 --gpus。用法见 python3 utils/release_GPU.py -h")

    leftover = release_processes(pids, ask=not args.yes)
    sys.exit(1 if leftover else 0)


if __name__ == "__main__":
    main()
