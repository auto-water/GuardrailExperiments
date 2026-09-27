# Benchmark 与 Guardrail 接口文档

本文以 `experiment_framework.py` 的现有接口和调用流程为准，规定新增 benchmark、guardrail 独立 Python 实现文件需要提供的逻辑。

## 1. 文件组织与职责

每个 benchmark 和每个 guardrail 分别放在独立的 `.py` 文件中。文件内定义对应的子类，以及该实现所需的数据路径、模型路径、读取或加载逻辑、格式适配逻辑和结果字段。

以下文件名仅用于说明组织方式，不代表已经实现或注册：

```text
codes/
├── experiment_framework.py        # 唯一实验入口、接口和通用执行逻辑
├── INTERFACES.md                  # 本文档
├── <benchmark_name>_benchmark.py  # 一个 benchmark 的完整实现
└── <guardrail_name>_guardrail.py  # 一个 guardrail 的完整实现
```

实现类继承 `Benchmark` 或 `Guardrail`。注册表保存能够无参数调用的实例创建方法；如果类本身可以无参数实例化，可以直接作为创建方法。需要自定义创建过程时，创建方法也放在对应实现文件中。

框架统一负责参数解析、GPU 选择与工作进程调度、样本分片、断点读取、实验循环、结果拼接、正常与异常记录、结果合并。实例通过接口提供差异逻辑。

模型加载放在 `Guardrail.load()` 中，数据读取放在 `Benchmark.read()` 中。模块导入和实例创建不启动实验，以便框架展示帮助及在各工作进程中创建实例。

## 2. 两侧共同遵守的数据约定

### 2.1 原始样例与统一样例

框架中的类型定义为：

```python
RawSample = dict[str, Any]


class Sample(TypedDict):
    content: str
    label: Literal["safe", "unsafe"]


Result = dict[str, Any]
```

- `RawSample`：benchmark 读取出的完整原始样例，字段结构由 benchmark 定义。内容应支持深拷贝和 JSON 序列化，以便保存完整异常样例。
- `Sample`：由 benchmark 组装，只包含 `content` 和 `label` 两个字段。
- `content`：待审核内容，必须是字符串；其具体构造格式由 benchmark 与 guardrail 配合约定。
- `label`：benchmark 提供的标准答案，只能是字符串 `"safe"` 或 `"unsafe"`。
- `Result`：两侧分别提供的结果字典，字段名为字符串，字段值须支持 JSON 序列化。

`Sample` 是接口约定；当前框架不会自动补齐字段、转换类型或执行通用样例校验。对应内容的解释和检查由具体实现完成。

benchmark 的原始标签如何转换为 `safe` / `unsafe`，由 benchmark 实现；护栏的预测类别如何映射为 `safe` / `unsafe`，由 guardrail 实现，包括多分类护栏的映射。

### 2.2 结果字段

两侧分别定义自己的结果字段，且不能重名。框架按以下方式拼接：

```python
result = {**benchmark_result, **guardrail_result}
```

拼接前框架检查字段冲突，冲突时中断实验。框架不会自动改名或覆盖重名字段，也不会自动把 `RawSample`、`Sample` 的字段加入正常结果。需要记录的内容应由对应的 `build_result()` 明确返回。

guardrail 的结果须包含映射后的分类结果与答案 token 的 top-2 概率；具体字段名称、概率结构及其他结果字段由实现定义。benchmark 的结果字段由其实现定义。

### 2.3 样本顺序与身份

样例 `id` 是 `list(benchmark.read())` 中从 0 开始的原始位置，由框架分配。两侧均不重新编号。

父进程和各工作进程会分别读取 benchmark；断点续跑也会重新读取。因此，`read()` 必须提供一致的样例数量、内容和顺序，不能自行按 GPU 分片、按断点过滤或随机重排。单条内容异常应留到逐样例接口中报告，以保留原始位置。

## 3. 新增 Benchmark 必须实现的逻辑

每个 benchmark 文件需提供一个 `Benchmark` 子类，实现以下 4 个接口。

| 接口 | 返回值 | 实现责任 |
| --- | --- | --- |
| `input_path`（属性） | `str` | 规定当前 benchmark 的输入路径，由实例提供，不从命令行读取。 |
| `read()` | `Iterable[RawSample]` | 从 `input_path` 读取完整原始样例，以字典逐条返回；顺序符合续跑要求。 |
| `adapt_sample(raw_sample: RawSample)` | `Sample` | 按与护栏约定的格式构造 `content`，将标准答案映射为 `safe` / `unsafe`，返回且仅返回这两个字段。 |
| `build_result(raw_sample: RawSample, sample: Sample)` | `Result` | 定义并组装 benchmark 侧要记录的结果字段，避免与护栏侧字段重名。 |

`build_result()` 在模型推理之前调用，只能使用原始样例与统一样例；当前接口不接收模型输出。

`adapt_sample()` 或 `build_result()` 发现单条样例内容有问题时，抛出 `BenchmarkSampleError`，报错信息应说明问题。文件无法读取、读取过程失败等非单条内容问题直接抛出原始异常。

### Benchmark 文件骨架

以下为未完成的接口骨架。方法体保留注释占位，需实现后才能注册和运行。

```python
from typing import Iterable

from experiment_framework import (
    Benchmark,
    BenchmarkSampleError,
    RawSample,
    Result,
    Sample,
)


class NewBenchmark(Benchmark):
    @property
    def input_path(self) -> str:
        """提供本实例的数据路径。"""
        # TODO: 返回数据输入路径。

    def read(self) -> Iterable[RawSample]:
        """按固定顺序提供全部原始样例。"""
        # TODO: 定义文件格式与读取逻辑，返回字典条目。

    def adapt_sample(self, raw_sample: RawSample) -> Sample:
        """提供 content 字符串及 safe/unsafe 标准答案。"""
        # TODO: 与护栏约定 content 格式。
        # TODO: 定义原始标签到 safe/unsafe 的映射。
        # TODO: 内容异常抛出 BenchmarkSampleError。

    def build_result(self, raw_sample: RawSample, sample: Sample) -> Result:
        """提供 benchmark 侧结果字段。"""
        # TODO: 定义并返回需要记录的字段。
        # TODO: 内容异常抛出 BenchmarkSampleError。
```

## 4. 新增 Guardrail 必须实现的逻辑

每个 guardrail 文件需提供一个 `Guardrail` 子类，实现以下 8 个接口。

| 接口 | 返回值 | 实现责任 |
| --- | --- | --- |
| `model_path`（属性） | `str` | 规定本实例的模型路径，由实例提供，不从命令行读取。 |
| `load()` | `None` | 加载模型、tokenizer 及本护栏所需资源；加载参数由实现定义。 |
| `adapt_input(sample: Sample)` | `Any` | 按与 benchmark 的约定解析 `content`，构造模型输入，包括本护栏所需的输入模板与编码。 |
| `infer(model_input: Any)` | `Any` | 执行一次推理，返回供后续解析、概率计算及完整 token 提取使用的原始输出。 |
| `get_answer_token_top2_probs(model_output: Any)` | `Any` | 定位答案 token，计算其 top-2 概率；定位规则、计算口径与返回结构由护栏定义。 |
| `get_output_tokens(model_output: Any)` | `Any` | 返回可 JSON 序列化的完整输出 token 序列，不截断；即使输出缺少分类等字段，也须能够提取。 |
| `build_result(model_output: Any, top2_probs: Any)` | `Result` | 解析预测，将分类结果映射为 `safe` / `unsafe`，并将计算好的 top-2 概率一起组织为护栏结果字段。 |
| `validate_output(model_output: Any, result: Result)` | `None` | 定义并检查本护栏必需的输出字段；输出缺字段等内容问题抛出 `ModelOutputError`。 |

### 4.1 输出适配与概率计算

`adapt_output()` 已有通用实现，新 guardrail 直接继承该实现：

```python
def adapt_output(self, model_output: Any) -> Result:
    top2_probs = self.get_answer_token_top2_probs(model_output)
    return self.build_result(model_output, top2_probs)
```

因此，具体护栏只需实现概率计算和结果构造接口。概率计算发生在输出适配内部，框架实验循环不单独计算概率。

`infer()` 返回的原始输出需要保留这两个接口和 `get_output_tokens()` 所需的信息。原始输出本身可以是护栏自定义的 Python 对象；写入结果或异常文件的字段必须转换为 JSON 可序列化内容。

答案缺失、输出不可解析等内容问题，在输出适配或校验阶段抛出 `ModelOutputError`。概率计算的实现错误、设备故障等直接抛出原始异常。

### 4.2 模型加载与设备

调度方通过 `CUDA_VISIBLE_DEVICES` 为每个工作进程限定一张 GPU，并设置 `DEVICE=cuda:0`。护栏加载逻辑应使用工作进程内的设备，不自行选择物理 GPU 或再启动实验工作进程。

模型路径、tokenizer 配置、模型加载参数、推理参数均在本护栏文件中定义；具体取值尚未统一规定。

### Guardrail 文件骨架

以下骨架不重复定义已有的 `adapt_output()`。其余方法需补充具体实现后再注册。

```python
from typing import Any

from experiment_framework import (
    BenchmarkSampleError,
    Guardrail,
    ModelOutputError,
    Result,
    Sample,
)


class NewGuardrail(Guardrail):
    @property
    def model_path(self) -> str:
        """提供本实例的模型路径。"""
        # TODO: 返回模型路径。

    def load(self) -> None:
        """加载模型及所需资源。"""
        # TODO: 定义加载配置，使用工作进程内的设备。

    def adapt_input(self, sample: Sample) -> Any:
        """将 content 适配为本护栏输入。"""
        # TODO: 按与 benchmark 的约定构造模型输入。
        # TODO: 样例内容异常抛出 BenchmarkSampleError。

    def infer(self, model_input: Any) -> Any:
        """执行推理并保留后续接口所需的原始输出。"""
        # TODO: 定义推理参数和调用逻辑。

    def get_answer_token_top2_probs(self, model_output: Any) -> Any:
        """提供答案 token 的 top-2 概率。"""
        # TODO: 定义答案位置、计算口径及返回结构。
        # TODO: 模型输出内容异常抛出 ModelOutputError。

    def get_output_tokens(self, model_output: Any) -> Any:
        """提供完整、可 JSON 序列化的输出 token 序列。"""
        # TODO: 从原始输出提取，不依赖分类等解析字段完整。

    def build_result(self, model_output: Any, top2_probs: Any) -> Result:
        """提供包含分类结果和 top-2 概率的护栏结果字段。"""
        # TODO: 定义类别到 safe/unsafe 的映射及结果字段。
        # TODO: 模型输出内容异常抛出 ModelOutputError。

    def validate_output(self, model_output: Any, result: Result) -> None:
        """检查本护栏规定的必需字段。"""
        # TODO: 定义必需字段及检查逻辑。
        # TODO: 模型输出内容异常抛出 ModelOutputError。
```

## 5. 固定调用顺序与异常边界

每个工作进程先调用 `guardrail.load()`，再完整读取 `benchmark.read()`，随后由记录器确定当前分片尚未完成的样例。每条待处理样例按以下顺序执行：

```text
保存原始样例的深拷贝
    → benchmark.adapt_sample(raw_sample)
    → benchmark.build_result(raw_sample, sample)
    → guardrail.adapt_input(sample)
    → guardrail.infer(model_input)
    → guardrail.adapt_output(model_output)
        → guardrail.get_answer_token_top2_probs(model_output)
        → guardrail.build_result(model_output, top2_probs)
    → guardrail.validate_output(model_output, guardrail_result)
    → 检查两侧结果字段无重名
    → 拼接并记录
```

异常是否能记录后继续，取决于异常类型及抛出阶段：

| 抛出阶段 | 可记录后继续的异常 | 框架行为 |
| --- | --- | --- |
| `adapt_sample()`、benchmark `build_result()`、`adapt_input()`、`infer()` | `BenchmarkSampleError`，仅用于单条样例内容问题 | 保存异常样例，计入完成进度，继续下一条。 |
| `adapt_output()` 内部的概率计算或结果构造、`validate_output()` | `ModelOutputError`，仅用于模型输出内容问题 | 调用 `get_output_tokens()`，保存异常输出，计入完成进度，继续下一条。 |
| 上述阶段中的其他异常，或异常类型出现在其他阶段 | 无 | 中断实验，不按内容异常跳过。 |
| 实例创建、模型加载、完整读取、深拷贝、token 提取、结果拼接、序列化或落盘 | 无 | 中断实验；未完整记录的当前样例不计入完成进度。 |

不要把所有异常统一包装成 `BenchmarkSampleError` 或 `ModelOutputError`。例如 `read()` 在开始逐条处理前被完整展开，在这里抛出 `BenchmarkSampleError` 也会中断；`infer()` 中抛出 `ModelOutputError` 同样不会进入输出异常记录分支。输出内容检查应放在输出适配或校验阶段。

## 6. 框架统一记录的内容

执行时，`-o` 指定输出目录。框架初始化两个异常子目录，并使用下列记录方式：

| 文件 | 内容 |
| --- | --- |
| `<guardrail_name>_<benchmark_name>.json` | 全部处理完成后的正常结果数组；元素是两侧结果字段的拼接，按原始样例顺序排列。 |
| `error_samples/<input_basename>.error.<id>.json` | 单条异常样例。 |
| `error_outputs/<input_basename>.output_error.<id>.json` | 单条异常模型输出。 |
| `<input_basename>.shard<shard_id>.jsonl` | 逐条写入的成功记录与样例索引，供断点续跑及结果合并使用。 |

`input_basename` 是 `benchmark.input_path` 的最后一个路径分量，包含原扩展名。最终结果文件的两个名称取自 `-g`、`-b` 所选择的注册名称。

异常样例记录包含：

| 字段 | 含义 |
| --- | --- |
| `source` | `-b` 指定的 benchmark 注册名称。 |
| `id` | 原始样例位置，从 0 开始。 |
| `error` | 异常类型及报错信息。 |
| `sample` | 开始处理前保存的完整原始样例。 |

异常输出还包含 `guardrail`（`-g` 指定的注册名称）和 `output_tokens`（护栏提供的完整 token 序列）。实例抛出异常并提供所需信息，文件由框架写入。

正常结果数组只包含成功样例。两类异常记录完整写入后，也计入已完成进度。若所有样例均为已记录的内容异常，最终正常结果为 `[]`。

续跑以 `-o` 目录中的旧分片和两类异常记录为依据；当前不检查 benchmark、guardrail 或数据版本是否匹配。该目录内的旧记录会按样例索引参与恢复。实例无需自行恢复断点。

如合并时仍有未完成样例，现有合并逻辑使用 `<guardrail_name>_<benchmark_name>.partial.json` 保存已有成功结果。运行故障中断时不执行最终合并。

## 7. 独立文件的注册与入口

框架当前维护两个空注册表：

```python
BENCHMARKS: dict[str, Callable[[], Benchmark]]
GUARDRAILS: dict[str, Callable[[], Guardrail]]
```

每个新实现完成后，需要把“注册名称 → 无参数实例创建方法”加入对应表。名称用于命令行选择、帮助列表，以及输出文件名或异常记录中的名称。绑定关系如下，仅为说明：

```text
BENCHMARKS[benchmark_name] → 对应 benchmark 文件中的创建方法或类
GUARDRAILS[guardrail_name] → 对应 guardrail 文件中的创建方法或类
```

**当前框架尚未实现独立实例模块的导入装配或自动发现。** 仅新增 `.py` 文件不会使其出现在 `-h` 中。后续接入须完成以下装配，当前保留为待实现事项：

```python
# TODO: 显式导入已完成的独立 benchmark、guardrail 实现。
# TODO: 在父进程和工作进程解析参数前，绑定名称与无参数创建方法。
# TODO: 确保实例导入的异常类与运行器捕获的异常类来自同一模块对象。
```

最后一项涉及当前直接执行脚本的方式：入口中的类属于 `__main__`，而独立文件用 `from experiment_framework import ...` 导入的类属于 `experiment_framework`。接入装配必须统一模块身份，否则同名的异常类可能不是同一个类，导致内容异常无法被运行器捕获。具体装配实现待后续接入时定义。

本文不创建实例，也不向当前注册表加入任何名称。完成实现及注册装配后，统一通过框架启动：

```text
python experiment_framework.py -h
python experiment_framework.py -b <benchmark_name> -g <guardrail_name> -o <output_directory>
```

命令行只提供 `-h`、`-b`、`-g`、`-o`。独立实例文件不提供额外实验入口或参数；数据和模型路径仍由对应实例声明。
