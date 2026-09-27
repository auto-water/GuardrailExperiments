"""护栏实验框架：加载护栏，遍历 benchmark，适配格式，推理并记录。"""

from abc import ABC, abstractmethod
import argparse
from copy import deepcopy
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Literal, Optional, TypedDict


# Benchmark 原始条目，交由 benchmark 实例组装样本和结果字段。
RawSample = dict[str, Any]


class Sample(TypedDict):
    """由 benchmark 实例组装并提供的统一样本，仅包含以下两个字段。"""

    content: str  # 待审核内容，具体格式由 benchmark 侧与护栏侧配合约定。
    label: Literal["safe", "unsafe"]  # 答案标签。


# 结果字段由 benchmark 侧和护栏侧分别提供，框架只拼接并记录。
# 两侧提供的结果字典须可被当前 JSON 记录方式序列化。
Result = dict[str, Any]


class BenchmarkSampleError(Exception):
    """由实例明确报告的 benchmark 样本内容问题，记录后继续下一条。"""


class ModelOutputError(Exception):
    """由护栏明确报告的模型输出内容问题，记录后继续下一条。"""


class Guardrail(ABC):
    """模型护栏接口；具体实现负责模型加载及模型侧的格式适配。"""

    @property
    @abstractmethod
    def model_path(self) -> str:
        """由当前护栏实例规定模型路径。"""
        # TODO: 在具体护栏实例中定义模型路径。

    @abstractmethod
    def load(self) -> None:
        """加载当前实例指定的护栏。"""
        # 调度方通过 CUDA_VISIBLE_DEVICES 限定一张卡，工作进程内 DEVICE=cuda:0。
        # TODO: 从当前实例的 model_path 加载护栏，具体加载参数待确定。

    @abstractmethod
    def adapt_input(self, sample: Sample) -> Any:
        """将 sample 中的 content 字符串适配为当前护栏接受的输入。"""
        # TODO: 与 benchmark 侧约定 content 的构造格式，并实现护栏输入适配。
        # 仅由样本内容导致的问题使用 BenchmarkSampleError；其他异常直接抛出。

    @abstractmethod
    def infer(self, model_input: Any) -> Any:
        """执行一次护栏推理，返回模型原始输出。"""
        # TODO: 实现具体推理调用，推理参数待确定。

    @abstractmethod
    def get_answer_token_top2_probs(self, model_output: Any) -> Any:
        """由护栏侧提供模型输出中答案 token 的 top-2 概率。"""
        # TODO: 由具体护栏定位答案 token，并计算其 top-2 概率。
        # TODO: 概率计算口径及返回结构由护栏侧定义。
        # 答案缺失等输出内容问题使用 ModelOutputError；计算或运行故障原样抛出。

    @abstractmethod
    def get_output_tokens(self, model_output: Any) -> Any:
        """提供可写入 JSON 的完整输出 token 序列，输出缺字段时也必须可用。"""
        # TODO: 由具体护栏从原始推理输出中提取完整 token 序列，不截断。

    def adapt_output(self, model_output: Any) -> Result:
        """在护栏侧计算答案 token 的 top-2 概率，并组装护栏结果字段。"""
        top2_probs = self.get_answer_token_top2_probs(model_output)
        return self.build_result(model_output, top2_probs)

    @abstractmethod
    def build_result(self, model_output: Any, top2_probs: Any) -> Result:
        """由护栏提供结果字段，包含 top-2 概率及映射后的分类结果。"""
        # TODO: 由具体护栏组织字段，并将分类结果映射为 safe 或 unsafe。
        # 输出内容不完整或不可解析时抛出 ModelOutputError；其他异常直接抛出。

    @abstractmethod
    def validate_output(self, model_output: Any, result: Result) -> None:
        """由护栏检查必需字段；输出缺字段等内容问题抛出 ModelOutputError。"""
        # TODO: 由具体护栏定义必需字段并检查原始输出和已解析结果。


class Benchmark(ABC):
    """Benchmark 接口；具体实现负责数据读取及数据侧的格式适配。"""

    @property
    @abstractmethod
    def input_path(self) -> str:
        """由当前 benchmark 实例规定数据输入路径。"""
        # TODO: 在具体 benchmark 实例中定义数据输入路径。

    @abstractmethod
    def read(self) -> Iterable[RawSample]:
        """读取全部原始样本，返回字典条目；续跑时须保持样本顺序一致。"""
        # TODO: 从当前实例的 input_path 读取 benchmark。

    @abstractmethod
    def adapt_sample(self, raw_sample: RawSample) -> Sample:
        """组装 content 字符串及 safe/unsafe 标签；内容问题抛出 BenchmarkSampleError。"""
        # TODO: 配合护栏侧约定的格式构造 content，并从原始条目提供 label。

    @abstractmethod
    def build_result(self, raw_sample: RawSample, sample: Sample) -> Result:
        """由当前 benchmark 实例提供需要记录的结果字段。"""
        # TODO: 由具体 benchmark 从原始条目和统一样本中组装其结果字段。
        # 仅由样本内容导致的问题使用 BenchmarkSampleError；其他异常直接抛出。


BENCHMARKS: dict[str, Callable[[], Benchmark]] = {
    # TODO: 注册已实现的 benchmark，名称对应无参数的实例创建方法。
}

GUARDRAILS: dict[str, Callable[[], Guardrail]] = {
    # TODO: 注册已实现的 guardrail，名称对应无参数的实例创建方法。
}


class GPUSelector:
    """沿用原选卡规则，查询空闲 GPU 并选择本次实验使用的卡。"""

    QUERY_FIELDS = "index,name,memory.used,memory.total,utilization.gpu"
    IDLE_MIB = 1024

    @staticmethod
    def _smi(args):
        try:
            return subprocess.run(
                ["nvidia-smi", *args], capture_output=True, text=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _to_int(value):
        try:
            return int(value)
        except ValueError:
            return None

    def show_smi(self):
        out = self._smi([])
        if out is None:
            print("没找到 nvidia-smi，无法枚举显卡", file=sys.stderr)
            return False
        print((out.stdout or out.stderr).rstrip())
        print()
        return out.returncode == 0

    def parse_gpus(self, text):
        """型号可以含逗号；未知数值保持为 None。"""
        gpus = []
        for line in text.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            index = self._to_int(parts[0])
            if index is None:
                continue
            gpus.append({
                "index": index,
                "name": ",".join(parts[1:-3]).strip(),
                "mem_used": self._to_int(parts[-3]),
                "mem_total": self._to_int(parts[-2]),
                "util": self._to_int(parts[-1]),
            })
        return gpus

    def query_gpus(self):
        out = self._smi([
            f"--query-gpu={self.QUERY_FIELDS}", "--format=csv,noheader,nounits",
        ])
        if out is None or out.returncode != 0:
            return []
        return self.parse_gpus(out.stdout)

    @staticmethod
    def free_mib(gpu):
        if gpu["mem_used"] is None or gpu["mem_total"] is None:
            return None
        return gpu["mem_total"] - gpu["mem_used"]

    def is_idle(self, gpu):
        return gpu["mem_used"] is not None and gpu["mem_used"] < self.IDLE_MIB

    def print_table(self, gpus):
        for gpu in gpus:
            free = self.free_mib(gpu)
            if free is None:
                mem, status = "显存未知", "未知"
            else:
                mem = f"显存 {gpu['mem_used']}/{gpu['mem_total']} MiB (空闲 {free} MiB)"
                status = "空闲" if self.is_idle(gpu) else "占用中"
            util = f"{gpu['util']}%" if gpu["util"] is not None else "未知"
            print(f"  [{gpu['index']}] {gpu['name']}   {mem}   利用率 {util}   {status}")

    def pick_gpus(self, candidates, count):
        def rank(gpu):
            free = self.free_mib(gpu)
            return (
                free is None, -(free if free is not None else 0),
                gpu["util"] if gpu["util"] is not None else 0,
            )

        return sorted(gpu["index"] for gpu in sorted(candidates, key=rank)[:count])

    @staticmethod
    def reject_reason(chosen, gpus, idle):
        by_index = {gpu["index"]: gpu for gpu in gpus}
        missing = [index for index in chosen if index not in by_index]
        if missing:
            return f"本机没有这些显卡编号 {missing}"
        busy = [index for index in chosen if index not in idle]
        if busy:
            detail = ", ".join(f"[{index}] 已用 {by_index[index]['mem_used']} MiB" for index in busy)
            return f"这些卡已被占用: {detail}"
        if len(chosen) > len(idle):
            return f"要 {len(chosen)} 张，但空闲卡只有 {len(idle)} 张"
        return None

    @staticmethod
    def ask_count(n_idle, n_total):
        if n_idle == 0:
            print("当前没有空闲显卡，无法启动", file=sys.stderr)
            return None
        while True:
            # TODO: 交互输入异常和中断处理暂不定义。
            raw = input(
                f"\n本次实验使用几张卡？（空闲 {n_idle} 张 / 共 {n_total} 张，直接回车=1）: "
            ).strip()
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

    def select_gpus(self, ask=True, count=None, indices=None):
        """沿用原工具的交互与选卡接口，不增加命令行参数。"""
        if not self.show_smi():
            return []
        gpus = self.query_gpus()
        if not gpus:
            print("nvidia-smi 没能枚举出任何显卡", file=sys.stderr)
            return []

        self.print_table(gpus)
        idle = {gpu["index"]: gpu for gpu in gpus if self.is_idle(gpu)}
        n_idle = len(idle)
        if indices is not None:
            chosen = sorted(set(indices))
        elif ask:
            count = self.ask_count(n_idle, len(gpus))
            if count is None:
                return []
            chosen = self.pick_gpus(list(idle.values()), count)
        else:
            if count is None:
                count = n_idle
            if n_idle == 0:
                print("当前没有空闲显卡，无法启动", file=sys.stderr)
                return []
            if not 1 <= count <= n_idle:
                print(f"要 {count} 张，但空闲卡只有 {n_idle} 张", file=sys.stderr)
                return []
            chosen = self.pick_gpus(list(idle.values()), count)

        reason = self.reject_reason(chosen, gpus, idle)
        if reason:
            print(f"选卡失败：{reason}", file=sys.stderr)
            return []

        by_index = {gpu["index"]: gpu for gpu in gpus}
        print(f"\n选用显卡: {chosen}")
        for index in chosen:
            gpu = by_index[index]
            free = self.free_mib(gpu)
            detail = f"空闲 {free} MiB" if free is not None else "显存未知"
            util = f"{gpu['util']}%" if gpu["util"] is not None else "未知"
            print(f"  [{index}] {gpu['name']}  {detail}  利用率 {util}")
        print(f"\n  export CUDA_VISIBLE_DEVICES={','.join(str(index) for index in chosen)}")
        return chosen


def _shard_paths(output_dir: str) -> list[str]:
    """以 -o 目录为准读取全部旧分片，不按模型、数据版本或输入文件名筛选。"""
    return sorted(glob.glob(os.path.join(output_dir, "*.shard*.jsonl")))


def _load_done_indices(shard_path: str) -> set[int]:
    """读取已完成的样本下标，沿用原实现对不完整记录的处理。"""
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
                continue
    return done


_ERROR_SUBDIRS = {
    "error": "error_samples",
    "output_error": "error_outputs",
}


def _initialize_output_directory(output_dir: str) -> None:
    """在指定输出目录中创建两类异常记录目录，重复初始化保留已有记录。"""
    for subdir in _ERROR_SUBDIRS.values():
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def _load_failed_indices(output_dir: str) -> set[int]:
    """两类异常 JSON 均作为已处理凭据，续跑时跳过对应的原始索引。"""
    done = set()
    for kind, subdir in _ERROR_SUBDIRS.items():
        for path in glob.glob(os.path.join(output_dir, subdir, f"*.{kind}.*.json")):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    done.add(json.load(f)["id"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    return done


class Recorder:
    """所有护栏共用的记录和续跑逻辑，迁自原 Shards 实现。

    单 GPU 和多 GPU 均使用分片 JSONL，逐条落盘，跳过已完成下标。
    已完整记录的失败样本和异常输出也计入已完成下标。
    没有 SHARD_ID 时默认一个分片，并在 close() 时合并。
    调度启动的工作进程由调度方等待全部结束后统一合并。
    """

    def __init__(
        self,
        items: list[RawSample],
        input_path: str,
        output_dir: str,
        result_filename: Optional[str] = None,
    ) -> None:
        self.items = items
        self.input_path = input_path
        self.output_dir = output_dir
        self.result_filename = result_filename
        _initialize_output_directory(output_dir)

        self.managed = "SHARD_ID" in os.environ
        if self.managed:
            self.num_shards = int(os.environ["NUM_SHARDS"])
            self.shard_id = int(os.environ["SHARD_ID"])
        else:
            self.num_shards = 1
            self.shard_id = 0
        self.shard_path = os.path.join(
            output_dir, f"{os.path.basename(input_path)}.shard{self.shard_id}.jsonl")

        # 扫描全部分片求并集，续跑时改变分片数也能跳过已完成样本。
        done = _load_failed_indices(output_dir)
        for path in _shard_paths(output_dir):
            done |= _load_done_indices(path)
        assigned = list(range(self.shard_id, len(items), self.num_shards))
        self.todo = [i for i in assigned if i not in done]
        self.desc = f"Processing shard {self.shard_id}/{self.num_shards}"
        self._fh = open(self.shard_path, "a", encoding="utf-8")
        print(f"[shard {self.shard_id}/{self.num_shards}] 分到 {len(assigned)} 条，其中 "
              f"{len(assigned) - len(self.todo)} 条已完成，本次待跑 {len(self.todo)} 条 "
              f"-> {self.shard_path}")

    def record(self, item_idx: int, item: Result) -> None:
        """所有运行方式均逐条追加并 flush、fsync。"""
        self._fh.write(json.dumps({"i": item_idx, "item": item}, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def record_error(
        self,
        source: str,
        item_idx: int,
        error: Exception,
        raw_sample: RawSample,
    ) -> None:
        """单独保存异常样本；文件完整写入后，该样本计入已完成进度。"""
        record = {
            "source": source,
            "id": item_idx,
            "error": f"{type(error).__name__}: {error}",
            "sample": raw_sample,
        }
        self._write_error_record("error", item_idx, record)

    def record_output_error(
        self,
        source: str,
        item_idx: int,
        error: Exception,
        raw_sample: RawSample,
        guardrail_name: str,
        output_tokens: Any,
    ) -> None:
        """记录异常输出的全部信息，并将该样本计入已完成进度。"""
        record = {
            "source": source,
            "id": item_idx,
            "error": f"{type(error).__name__}: {error}",
            "sample": raw_sample,
            "guardrail": guardrail_name,
            "output_tokens": output_tokens,
        }
        self._write_error_record("output_error", item_idx, record)

    def _write_error_record(self, kind: str, item_idx: int, record: dict[str, Any]) -> None:
        error_path = os.path.join(
            self.output_dir, _ERROR_SUBDIRS[kind],
            f"{os.path.basename(self.input_path)}.{kind}.{item_idx}.json",
        )
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def close(self) -> None:
        """关闭分片文件；独立运行时同时合并，调度模式等待父进程合并。"""
        self._fh.close()
        print(f"[shard {self.shard_id}/{self.num_shards}] 完成，本次处理 {len(self.todo)} 条 "
              f"-> {self.shard_path}")
        if not self.managed:
            self.merge()

    def abort(self) -> None:
        """运行故障时关闭记录文件，保留已有断点，不宣布完成或合并结果。"""
        self._fh.close()

    def merge(self) -> bool:
        return self.merge_shards(
            self.items, self.input_path, self.output_dir, self.result_filename,
        )

    @staticmethod
    def merge_shards(
        items: list[RawSample],
        input_path: str,
        output_dir: str,
        result_filename: Optional[str] = None,
    ) -> bool:
        """合并成功结果，以成功记录和异常记录的并集判断是否全部处理完成。"""
        total = len(items)
        merged = {}
        for path in _shard_paths(output_dir):
            n = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    merged[rec["i"]] = rec["item"]
                    n += 1
            print(f"[merge] {os.path.basename(path)}: {n} 条")

        failed = _load_failed_indices(output_dir)
        if not merged and not failed:
            print("[merge] 没有找到任何分片文件，什么都没写")
            return False

        out = [merged[i] for i in range(total) if i in merged]
        done = set(merged) | failed
        missing = [i for i in range(total) if i not in done]
        base = result_filename if result_filename is not None else os.path.basename(input_path)
        if missing:
            out_path = os.path.join(output_dir, os.path.splitext(base)[0] + ".partial.json")
            print(f"[merge] 还差 {len(missing)}/{total} 条（例如下标 {missing[:5]}），"
                  "说明还有分片没跑完，重跑同一条命令会从断点继续")
        else:
            out_path = os.path.join(output_dir, base)
            print(f"[merge] {total} 条全部处理完成，正常结果 {len(out)} 条，"
                  f"异常记录 {sum(i in failed and i not in merged for i in range(total))} 条")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[merge] 已写入 {out_path}")
        return not missing


class ExperimentRunner:
    """接收指定的具体实现，执行固定的完整实验流程。"""

    def __init__(
        self,
        guardrail: Guardrail,
        benchmark: Benchmark,
        output_dir: str,
        benchmark_name: str,
        guardrail_name: str,
    ) -> None:
        self.guardrail = guardrail
        self.benchmark = benchmark
        self.input_path = benchmark.input_path
        self.output_dir = output_dir
        self.benchmark_name = benchmark_name
        self.guardrail_name = guardrail_name
        self.result_filename = f"{guardrail_name}_{benchmark_name}.json"
        _initialize_output_directory(output_dir)

    def run(self) -> None:
        """处理本次待跑样本，拼接 benchmark 和护栏提供的结果字段并记录。"""
        self.guardrail.load()
        raw_samples = list(self.benchmark.read())
        self.recorder = Recorder(
            raw_samples, self.input_path, self.output_dir, self.result_filename,
        )

        try:
            for item_idx in self.recorder.todo:
                raw_sample = raw_samples[item_idx]
                original_sample = deepcopy(raw_sample)
                try:
                    sample = self.benchmark.adapt_sample(raw_sample)
                    benchmark_result = self.benchmark.build_result(raw_sample, sample)
                    model_input = self.guardrail.adapt_input(sample)
                    model_output = self.guardrail.infer(model_input)
                except BenchmarkSampleError as error:
                    self.recorder.record_error(
                        self.benchmark_name, item_idx, error, original_sample,
                    )
                    continue

                try:
                    guardrail_result = self.guardrail.adapt_output(model_output)
                    self.guardrail.validate_output(model_output, guardrail_result)
                except ModelOutputError as error:
                    output_tokens = self.guardrail.get_output_tokens(model_output)
                    self.recorder.record_output_error(
                        self.benchmark_name, item_idx, error, original_sample,
                        self.guardrail_name, output_tokens,
                    )
                    continue

                duplicate_fields = benchmark_result.keys() & guardrail_result.keys()
                if duplicate_fields:
                    raise ValueError(
                        "benchmark 与 guardrail 的结果字段不能重名："
                        + ", ".join(sorted(duplicate_fields))
                    )
                result = {**benchmark_result, **guardrail_result}
                self.recorder.record(item_idx, result)
        except Exception:
            self.recorder.abort()
            raise

        self.recorder.close()


class ExperimentScheduler:
    """从框架入口启动工作进程，一张 GPU 对应一个分片，结束后统一合并。"""

    def __init__(
        self,
        benchmark_name: str,
        guardrail_name: str,
        benchmark: Benchmark,
        output_dir: str,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.guardrail_name = guardrail_name
        self.result_filename = f"{guardrail_name}_{benchmark_name}.json"
        self.benchmark = benchmark
        self.input_path = benchmark.input_path
        self.output_dir = output_dir
        _initialize_output_directory(output_dir)

    def _launch(self, gpus: list[int]):
        base = os.path.basename(self.input_path)
        procs = []
        try:
            for shard_id, gpu in enumerate(gpus):
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["NUM_SHARDS"] = str(len(gpus))
                env["SHARD_ID"] = str(shard_id)
                env["DEVICE"] = "cuda:0"
                log_path = os.path.join(self.output_dir, f"{base}.shard{shard_id}.log")
                log = open(log_path, "w", encoding="utf-8")
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "-u", os.path.abspath(__file__),
                         "-b", self.benchmark_name, "-g", self.guardrail_name,
                         "-o", self.output_dir],
                        env=env, stdout=log, stderr=subprocess.STDOUT,
                    )
                except Exception:
                    log.close()
                    raise
                procs.append((shard_id, gpu, log, proc))
                print(f"[shard {shard_id}] GPU {gpu} 启动，pid {proc.pid}，日志 {log_path}")
        except Exception:
            self._stop_workers(procs)
            raise
        return procs

    @staticmethod
    def _collect(procs) -> None:
        pending = list(procs)
        while pending:
            for worker in pending[:]:
                shard_id, gpu, log, proc = worker
                returncode = proc.poll()
                if returncode is None:
                    continue
                log.close()
                pending.remove(worker)
                print(f"[shard {shard_id}] GPU {gpu} 进程结束，退出码 {returncode}，日志 {log.name}")
                if returncode != 0:
                    raise RuntimeError(
                        f"工作进程 {shard_id} 发生运行故障，退出码 {returncode}，见 {log.name}"
                    )
            if pending:
                time.sleep(0.1)

    @staticmethod
    def _stop_workers(procs) -> None:
        """只停止本次实验启动的工作进程，防止运行故障后其他分片继续执行。"""
        for _, _, _, proc in procs:
            if proc.poll() is None:
                proc.kill()
        for _, _, log, proc in procs:
            proc.wait()
            log.close()

    def run(self, gpus: list[int]) -> bool:
        items = list(self.benchmark.read())
        base = os.path.basename(self.input_path)
        with open(os.path.join(self.output_dir, f"{base}.manifest.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"input_path": self.input_path, "num_shards": len(gpus),
                 "gpus": gpus, "total": len(items)},
                f, ensure_ascii=False, indent=2,
            )
        # TODO: 用户主动中断的处理暂不定义。
        procs = self._launch(gpus)
        try:
            self._collect(procs)
        except Exception:
            self._stop_workers(procs)
            raise
        return Recorder.merge_shards(
            items, self.input_path, self.output_dir, self.result_filename,
        )


# TODO: 根据后续指定的模型护栏，实现 Guardrail 子类。
# TODO: 根据后续指定的 benchmark，实现 Benchmark 子类。


def main() -> None:
    """实验唯一命令行入口，仅接受 -h、-b、-g、-o。"""
    benchmark_names = ", ".join(sorted(BENCHMARKS)) or "暂无（未注册）"
    guardrail_names = ", ".join(sorted(GUARDRAILS)) or "暂无（未注册）"
    parser = argparse.ArgumentParser(
        description="选择 benchmark 和 guardrail，运行护栏实验。",
        add_help=False,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(f"可用 benchmark：{benchmark_names}\n"
                f"可用 guardrail：{guardrail_names}"),
    )
    parser.add_argument(
        "-h", action="help",
        help="列出全部参数功能及可用的 benchmark、guardrail，然后退出。",
    )
    parser.add_argument(
        "-b", dest="benchmark", metavar="name", required=True,
        choices=BENCHMARKS, help="指定 benchmark 名称（必填）。",
    )
    parser.add_argument(
        "-g", dest="guardrail", metavar="name", required=True,
        choices=GUARDRAILS, help="指定 guardrail 名称（必填）。",
    )
    parser.add_argument(
        "-o", dest="output_dir", metavar="directory", required=True,
        help="指定实验输出目录（必填）。",
    )
    args = parser.parse_args()

    benchmark_factory = BENCHMARKS[args.benchmark]
    guardrail_factory = GUARDRAILS[args.guardrail]
    if "SHARD_ID" in os.environ:
        runner = ExperimentRunner(
            guardrail=guardrail_factory(),
            benchmark=benchmark_factory(),
            output_dir=args.output_dir,
            benchmark_name=args.benchmark,
            guardrail_name=args.guardrail,
        )
        runner.run()
        return

    gpus = GPUSelector().select_gpus()
    if not gpus:
        sys.exit(1)
    scheduler = ExperimentScheduler(
        benchmark_name=args.benchmark,
        guardrail_name=args.guardrail,
        benchmark=benchmark_factory(),
        output_dir=args.output_dir,
    )
    if not scheduler.run(gpus):
        sys.exit(1)


if __name__ == "__main__":
    main()
