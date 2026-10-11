# wan_parallel

基于固定版本 Wan2.1-T2V-1.3B 的 TP2 / SP2 推理工程。本目录是新工程根目录，Python 包和后续入口均在本目录组织。

## 项目结构

```text
wan_parallel/
├── parallel/
│   ├── __init__.py
│   ├── tp_linear.py          # 已有实现迁移，尚未执行验证
│   ├── tp_weight_loader.py   # TP 分片与复制参数加载，尚未验证
│   ├── tp_attention.py       # T2V Self/CrossAttention TP，尚未验证
│   └── sp_attention.py       # 占位
├── models/
│   ├── __init__.py
│   └── wan/
│       ├── __init__.py
│       └── tp_wan.py         # meta 构造、Attention/FFN TP 替换与完整 DiT 参数加载
├── runtime/
│   ├── __init__.py
│   ├── distributed.py        # torchrun GPU 绑定、WORLD 初始化与销毁，尚未验证
│   └── parallel_state.py     # TP 组创建、获取与销毁，尚未验证
├── pipelines/
│   ├── __init__.py
│   └── wan_t2v_tp.py          # TP 组件初始化、输入广播与官方语义采样，尚未验证
├── scripts/
│   └── generate.py           # torchrun TP 生成、rank 0 保存与进程组清理，尚未验证
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_wan_single_gpu.py  # 官方 Wan 单卡基线测量入口
│   └── benchmark_wan_tp2.py         # TP2 双卡性能测量，尚未执行
├── .gitignore
├── .gitattributes
├── README.md
└── pyproject.toml
```

本仓库保存项目源码、单卡基线测量入口及必要运行说明；官方源码、模型权重、虚拟环境、运行结果和本地规划记录独立保存，不随本仓库发布。官方 Wan 参考 commit 为 `9737cba9c1c3c4d04b33fcad41c111989865d315`。本工程目前不是包含完整官方依赖的独立发行包。

## 云端目录组织

两个源码仓库保持独立，共用一个已准备好完整 Wan 依赖的 Python 环境：

```text
/root/autodl-tmp/wan_xdit/
├── .venv/                   # 共用的运行环境
├── Wan2.1/                  # 官方源码仓库
│   └── wan/
├── wan_parallel/            # 从 GitHub 下载的本项目
│   ├── parallel/
│   ├── models/
│   ├── pipelines/
│   ├── runtime/
│   ├── scripts/
│   ├── benchmarks/
│   └── pyproject.toml
├── models/
│   └── Wan2.1-T2V-1.3B/     # 权重目录
└── outputs/
```

从 `wan_parallel` 工程根目录启动。`models/wan/` 是本项目的 Python 模型适配包，外层 `../models/Wan2.1-T2V-1.3B/` 才是权重目录，两者职责不同。官方源码也可以放在其他位置，生成入口通过 `PYTHONPATH` 查找 `wan`，基线入口通过 `--wan-dir` 指定源码位置。本地研究工作区仍将官方参考源码保存在外层 `../third_party/Wan2.1/`，该布局不要求随 GitHub 仓库发布。

## 职责与接入流程

- `parallel/`：参数分片、本地计算与通信，以及 checkpoint 分片读取。
- `models/wan/tp_wan.py`：组织 meta 构造、Attention/FFN TP 替换、存储分配和 RoPE 频率重建，保留原始计算语义。
- `runtime/distributed.py`：绑定 torchrun 对应 GPU，初始化和销毁默认 NCCL 进程组 WORLD。
- `runtime/parallel_state.py`：按显式 TP size 创建、获取和销毁 TP 组；由运行入口组织其生命周期。
- `pipelines/wan_t2v_tp.py`：组织官方 T5/VAE、TP DiT 构造与加载、输入广播和采样；返回 rank 0 视频。
- `scripts/generate.py`：命令行参数、WORLD/TP 生命周期、管线调用与 rank 0 视频保存。
- `benchmarks/benchmark_wan_single_gpu.py`：复用原始官方管线，观测单卡时间与显存，作为 TP/SP 的基线。
- `benchmarks/benchmark_wan_tp2.py`：调用现有 TP 管线，记录逐 rank 数据和双卡最大值；可选读取匹配单卡结果计算加速比。

已确定的接入顺序：

```text
create_wan_tp_model(config, process_group, device=..., dtype=torch.float32)
  ├─ 在 meta 上通过官方 WanModel.from_config 创建模型
  ├─ parallelize_wan_model(..., device="meta")，统一替换 FFN 与 Attention
  ├─ to_empty() 为最终分片结构分配存储
  └─ 按官方公式重建 freqs，保留 complex128
  → load_wan_tp_weights() 加载 TP 分片及剩余复制参数
  → WanModel.forward()
```

`create_wan_tp_model()` 接收官方 T2V 配置字典及已经初始化的进程组，返回已分配存储、尚未加载权重的模型。完整 FFN 和 Attention 参数只在 meta 上存在，不会先分配完整参数数据。默认使用 FP32 参数，对齐现有 BF16 autocast 基准。构造入口不读取 checkpoint，也不初始化进程组。

当前替换 FFN 的两个 Linear，以及每个 block 的 SelfAttention 和 T2V CrossAttention，保留原 GELU、block Norm、调制与残差。Attention 按完整 heads 切分 Q/K/V，Q/K RMSNorm 归约全 hidden 维统计，Row 输出投影归约后加一次 bias。SelfAttention 复用官方 RoPE 和视频 K/V 长度处理；CrossAttention 不使用 RoPE，原样传递 `context_lens`。统一替换入口仅支持 T2V，并在一次 block 遍历中完成三部分替换。

`load_wan_tp_weights()` 在替换及存储分配后调用，先遍历 blocks 加载 FFN 和 self/cross attention 分片，再遍历实际注册的模块加载剩余复制参数。加载链为：

```text
load_wan_tp_weights
  ├─ load_linear_weight：ffn.0、ffn.2
  ├─ load_attention_weights：self_attn、cross_attn
  │    ├─ load_linear_weight：q、k、v、o
  │    └─ qk_norm 启用时，load_rmsnorm_weight：norm_q、norm_k
  └─ load_replicated_weights：其余完整参数
```

RMSNorm 读取 `blocks.<i>.<self_attn/cross_attn>.norm_q.weight` 和 `norm_k.weight`。TP2 将完整 `[1536]` affine 权重按组内 rank 切为 `[768]`，与 Q/K 投影的通道范围一致，原地填充已有参数；RMSNorm 没有 bias。

复制参数 loader 遍历每个模块的直属参数，跳过 TP Linear/RMSNorm；Row Linear 的完整 bias 已由 Linear loader 加载，也不会重复读取。剩余参数涵盖 patch/text/time embedding、time projection、block modulation 与带 affine 的 Norm、输出 Head。两卡从同一 checkpoint 逐张量在 CPU 读取，检查精确形状并转换为目标 dtype，再原地填充已有参数，不一次性构造完整 GPU state dict。无 affine 的 Norm 和 Identity 没有参数，不需额外分支；`freqs` 不属于注册参数，继续按官方公式重建。

构造与完整 DiT 参数加载链已编写，实际模型加载和 forward 尚未在云端验证。官方模型源码保持原样。SP 后续单独推进。

调用方需要让固定版本官方源码仓库根目录（上述布局中的 `../Wan2.1`）位于 Python 模块搜索路径中，并具备官方 Wan 运行依赖；适配器通过 `wan.modules.model` 导入 `WanModel` 和 `rope_params`，不在适配器代码中修改 `sys.path`。

## TP 管线组件初始化

`pipelines.wan_t2v_tp.WanT2VTP` 是项目自有管线，不继承或实例化官方 `WanT2V`。初始化调用关系为：

```text
运行入口初始化 WORLD / TP，并绑定当前 CUDA device
  → WanT2VTP(config, checkpoint_dir, process_group=tp_group, device=device)
      ├─ TP 组内 rank 0：创建官方 CPU T5 和当前 GPU 上的官方 VAE
      ├─ 所有 TP ranks：读取 checkpoint_dir/config.json
      ├─ create_wan_tp_model(..., dtype=torch.float32)
      ├─ load_wan_tp_weights(..., diffusion_pytorch_model.safetensors)
      └─ model.eval().requires_grad_(False)
```

构造函数接收官方 `WAN_CONFIGS` 中的管线配置，T5/VAE 路径和精度从该配置或官方组件默认值取得；DiT 结构单独读取 checkpoint 的 `config.json`，交给现有 `WanModel.from_config` 路径处理。配置不能混用。当前采用准备阶段记录的固定 checkpoint 布局，不增加其他模型格式或 checkpoint 分片索引的兼容路径。

DiT 在 meta 上完成 TP 模块替换，再在当前 GPU 分配最终参数存储并加载权重；不调用 `WanModel.from_pretrained`，不先加载完整 DiT，也不改变已有 TP 权重加载规则。参数保持 FP32；BF16 autocast 属于后续采样路径，不将管线配置中的 `param_dtype` 当作参数转换指令。

T5/VAE 只由 TP 组内 rank 0 持有，其他 rank 的 `text_encoder` 和 `vae` 为 `None`。这是相对官方每个 rank 创建组件的资源组织调整，后续性能比较需要记录。T5 始终先在 CPU 构造，`t5_cpu=True` 为默认值，对齐正式单卡基准；该标志用于后续编码阶段的放置选择，不在构造阶段搬运 T5。VAE 沿用官方默认 FP32 缩放常量与 decode 设置；其权重通过 `assign=True` 加载，参数精度由 checkpoint 决定，加载后实际 dtype 尚待确认。

运行入口拥有 WORLD/TP 生命周期；管线只保存传入的 TP group 和 device，不初始化、销毁进程组，不重复保存 rank/size。调用方保证所有 rank 使用同一 checkpoint 和配置，并在管线使用期间保持组存活。构造函数没有新增 barrier；初始化中途失败直接暴露，清理由运行入口负责，不增加错误清理 barrier。

初始化与生成由构造函数和 `generate` 分别负责。latent 通道数使用已加载 DiT 的 `in_dim`，空间与时间压缩使用管线配置的 `vae_stride`，patch 大小使用模型的 `patch_size`；非零 rank 无需访问 VAE。当前没有执行组件加载或其他云端验证，本机仅编辑与人工审阅。

## TP 采样与输入契约

所有 TP ranks 共同调用 `WanT2VTP.generate`，使用相同的 `size=(width, height)`、`frame_num`、`sample_solver`、`sampling_steps`、`shift`、`guide_scale` 和 `offload_model`。调用方负责保持这些参数以及构造配置一致；管线不重复广播配置或逐步检查状态。TP 组内 rank 0 的正负提示词和 seed 是生成输入的权威来源。

```text
各 rank 根据相同参数创建独立的官方 scheduler / timesteps
  → rank 0 确定 seed，广播给全组
  → rank 0 按官方 CPU/GPU T5 路径编码正负提示词
  → 分别广播真实文本长度和 embedding
  → rank 0 生成初始 FP32 noise，广播给全组
  → 所有 ranks 将 DiT 放到当前 GPU
  → 每个 timestep：conditional TP forward → unconditional TP forward
      → CFG 合成 → 本地 scheduler.step → 下一步完整 latent
  → 按 offload_model 将 DiT 移到 CPU
  → rank 0 官方 VAE decode
  → TP 组完成 barrier
  → rank 0 返回 [3,F,H,W] 视频，其他 rank 返回 None
```

文本广播用一个私有 `_broadcast_context` 方法处理两种提示词。接收端先获取 token 长度，再按官方 T5 输出 dtype（`config.t5_dtype`）和 DiT 的 `text_dim` 分配张量。官方 T5 最终归一化返回该配置精度；不额外转换 embedding 精度，也不在广播前补到 512。后续仍由官方 DiT forward 补零并保持原 cross-attention 语义。NCCL 通信张量在当前 GPU 上，广播源通过 TP 组内 rank 0 映射到全局 rank。

采样复用固定 Wan commit 的 `FlowUniPCMultistepScheduler` 和 `FlowDPMSolverMultistepScheduler`，保留 shift/timesteps 建立方法、CFG 公式和 scheduler 输入的 batch 维转换。DPM++ 使用官方默认确定性 `dpmsolver++`，没有扩展 SDE 配置。每次生成新建 scheduler，不跨调用保存历史；每个 rank 始终顺序执行两次 TP forward，不进行 CFG 并行。相同初始输入、时间序列与归约预测是各 rank 状态一致的条件，目前仍待云端确认；不增加每步 latent 广播或一致性检查。

`generate` 的参数默认值沿用官方方法（1280×720、81 帧、50 steps、shift=5、CFG=5、UniPC、seed=-1、offload=True），正式基准应显式传入 832×480、81 帧、50 steps、shift=8、CFG=6、UniPC、seed=42，并在构造时使用 `t5_cpu=True`。DiT FP32 参数与配置中的 BF16 autocast 分开；VAE 精度与缩放继续由官方组件控制。

DiT 仅在每次调用的循环前搬回 GPU，在整个采样循环结束后 offload；CPU T5 路径保持文本编码后将 embedding 搬到 GPU。GPU T5 路径在编码后按 `offload_model` 决定是否搬回 CPU。原官方循环内重复的 `model.to(device)` 移到循环前，避免重复模块遍历；计算顺序不变。当前不使用 FSDP，因此不迁移官方 `no_sync` 包装。只有 rank 0 显示采样进度。

正常生成结束保留 TP barrier，避免其他 rank 在 rank 0 解码完成前开始下一次调用；异常直接传播，错误清理不加入 barrier 或自动恢复。视频保存和 WORLD/TP 的 finally 清理由 `scripts/generate.py` 负责。

初始化、文本/noise 广播、两种 solver、重复调用后的 GPU 恢复、跨 rank 状态一致性及 VAE 解码均尚未执行云端验证。本轮只进行源码编辑与人工审阅，没有运行本地测试或新增验证脚本。

## TP 生成入口

`scripts/generate.py` 面向固定版本 T2V-1.3B，使用 torchrun 提供的 rendezvous 和 rank 环境变量。入口先绑定当前 GPU、初始化 WORLD/TP，再导入官方 Wan 和管线，避免官方模块导入时查询了错误的当前 CUDA device。所有 ranks 从同一启动命令取得采样参数；TP size 由 `--tp-size` 显式配置，runtime 要求其与 WORLD size 一致，不在入口重复校验该条件。

CLI 默认对齐正式单卡 workload：832×480、81 帧、50 steps、UniPC、shift=8、CFG=6、seed=42、CPU T5、offload=True，使用官方 T2V-1.3B 示例 prompt 和默认负提示词。默认值区别于管线 `generate` 的官方通用默认值；入口显式传递全部生成参数，不改变管线接口。`--size` 按官方 `SUPPORTED_SIZES` / `SIZE_CONFIGS` 解析，目前为 `832*480` 或 `480*832`；帧数须为正的 4n+1。

| 参数 | 默认值 / 含义 |
|---|---|
| `--ckpt-dir` | 必填，checkpoint 目录 |
| `--tp-size` | 2，WORLD 成员数必须相等 |
| `--size` / `--frame-num` | `832*480` / 81 |
| `--sample-solver` / `--sample-steps` | `unipc` / 50；solver 可选 `dpm++` |
| `--sample-shift` / `--guide-scale` | 8 / 6 |
| `--seed` | 42；负值由管线 rank 0 随机选 seed |
| `--prompt` / `--negative-prompt` | 官方示例 / 空字符串使用官方负提示词 |
| `--t5-cpu` / `--no-t5-cpu` | 默认 CPU；关闭后在 GPU 编码 |
| `--offload-model` / `--no-offload-model` | 默认开启；关闭后 DiT 留在 GPU |
| `--save-file` | `outputs/wan_tp.mp4`，相对当前工程目录 |
| `--timeout-seconds` | 1800，WORLD/TP 通信超时，包含等待加载与解码 |

rank 0 创建输出父目录并调用官方 `cache_video`，保留 `fps=config.sample_fps`、`nrow=1`、`normalize=True`、`value_range=(-1,1)` 和官方 MP4 编码方式。保存采用一次尝试（`retry=1`）；官方工具失败时打印原因并返回 `None`，入口据此抛出错误，避免保存失败仍报告任务成功。同一路径已有文件时按官方 writer 行为写入，需保留结果时为每次生成指定不同文件名。

管线末尾 barrier 覆盖解码，入口在保存后还有一个 TP barrier，覆盖文件输出；只有全组到达该正常完成边界后才报告完成。`finally` 始终先销毁 TP，再销毁 WORLD，即使 TP 清理抛错仍尝试 WORLD 清理；异常清理没有 barrier、自动重试或恢复分支。rank 0 保存失败时不进入正常完成 barrier，其他 rank 可能等待到超时，跨 rank 故障终止仍依赖 torchrun。

云端启动方式（以下为生成入口使用示例，尚未执行，不代表验证通过）：

```bash
cd /root/autodl-tmp/wan_xdit/wan_parallel
export PYTHONPATH="$PWD:$PWD/../Wan2.1"
../.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  --module scripts.generate \
  --ckpt-dir ../models/Wan2.1-T2V-1.3B \
  --tp-size 2 \
  --save-file ../outputs/wan_tp2_f81_s50_seed42.mp4
```

入口通过工程根目录下的 `scripts.generate` 模块启动；`PYTHONPATH` 提供项目包和固定版本官方 `wan` 包，不在 Python 代码中修改 `sys.path`。使用云端现有 `.venv` 及完整官方依赖（包括视频编码依赖），本项目的 `pyproject.toml` 不替代官方运行环境。该入口只生成和保存视频，不包含正确性比较、性能测量或自动化验证。

## 官方单卡基线测量

`benchmarks/benchmark_wan_single_gpu.py` 由原 `scripts/benchmark_single_gpu.py` 迁入并按用途重命名，测量逻辑保持原样。它调用固定 commit 的官方 `WanT2V`，使用 hooks 记录每次 DiT forward 的 CUDA Events，用前后 GPU 同步的墙钟计时记录完整 generate，并记录逐轮显存峰值。它不使用本项目的 TP 管线，也不是单元测试。

脚本从本工程根目录启动，默认官方源码位置为 `../Wan2.1`，权重位置为 `../models/Wan2.1-T2V-1.3B`；支持通过 `--wan-dir` 和 `--ckpt-dir` 指定其他位置。可选模型 revision 记录从 `../env-info/wan-model-revision.txt` 读取。默认小工作负载仍为 17 帧 / 5 steps；正式基线使用显式的 81 帧 / 50 steps。CPU 线程设置也需与已有基线的 intra-op threads=1 对齐。

```bash
cd /root/autodl-tmp/wan_xdit/wan_parallel
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ../.venv/bin/python \
  benchmarks/benchmark_wan_single_gpu.py \
  --wan-dir ../Wan2.1 \
  --ckpt-dir ../models/Wan2.1-T2V-1.3B \
  --frames 81 --steps 50 --seed 42 \
  --warmup 1 --runs 3 \
  --t5-cpu --offload-model \
  --output-dir ../outputs/baseline/wan_single_gpu_f81_s50_seed42
```

上述迁移后的启动方式尚未执行。每轮结果写入指定输出目录的 `metrics.json`，最后一轮视频为 `last_run.mp4`；已有 metrics 的输出目录拒绝覆盖。一次预热和三次正式测量的统计规则不变，`complete=true` 表示全部测量、保存与汇总结束。单卡生成时间包含 T5、sampling、offload、VAE 和 cleanup，排除模型加载与 MP4 编码；DiT CUDA 区间合计不是完整 denoising 时间。测量记录实际参数/autocast dtype、attention backend、GPU 和环境信息。

运行结果、日志和视频只保存在云端或本地研究工作区，不提交到本源码仓库。既有正式单卡结果保持保留；TP 正确性对比和 SP 性能测量入口尚未实现。

## TP2 双卡性能测量

`benchmarks/benchmark_wan_tp2.py` 面向单机 TP=2、两个 GPU，以 torchrun 启动两个进程；`--tp-size` 只接受 2，WORLD size 与 TP size 的一致性沿用 runtime 检查。它调用现有 `WanT2VTP.generate`，不复制采样循环或修改 TP collective。默认采用正式 workload：832×480、81 帧、50 steps、UniPC、shift=8、CFG=6、seed=42、CPU T5、offload=True，预热 1 次、正式测量 3 次。可显式传入 `--frames 17 --steps 5` 先测小 workload，结果须使用新的输出目录，并与对应的小 workload 单卡结果比较。

以下云端命令待执行；本机没有运行语法检查、import 检查、CLI、CPU/GPU 测试或测量：

```bash
cd /root/autodl-tmp/wan_xdit/wan_parallel
export PYTHONPATH="$PWD:$PWD/../Wan2.1"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 ../.venv/bin/torchrun \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --module benchmarks.benchmark_wan_tp2 \
  --wan-dir ../Wan2.1 \
  --ckpt-dir ../models/Wan2.1-T2V-1.3B \
  --tp-size 2 \
  --frames 81 --steps 50 --seed 42 \
  --warmup 1 --runs 3 \
  --t5-cpu --offload-model \
  --output-dir ../outputs/benchmarks/wan_tp2_f81_s50_seed42
```

`--wan-dir` 用于确认版本及源码记录，实际 `wan` 模块仍由 `PYTHONPATH` 提供；两者不一致时拒绝测量，避免把另一份源码的版本信息记入报告。固定 Wan commit 检查与单卡基线相同。输出为 rank 0 写入的 `metrics.json` 和最后一轮 `last_run.mp4`，拒绝覆盖已有 metrics；失败时保留 `complete=false` 的部分报告，只有全部测量、视频保存、汇总及可选对比完成后写入 `complete=true`。

每轮在计时前完成 GPU 同步和 TP barrier，对齐开始边界；随后记录各 rank 从 generate 调用到返回并完成 GPU 同步的本地墙钟时间。管线自身的完成 barrier 已包含 rank 0 解码等待。每个 rank 在自己的 CUDA device 上用事件记录顺序的 `2×steps` 次 DiT forward，CPU 数据读取、视频形状/有限值检查、结果 gather、JSON 和 MP4 写入均在计时外。

| 报告字段 | 含义 |
|---|---|
| `runs[].ranks` / `warmup_results[].ranks` | 每个 rank 的原始时间、forward 区间数组、显存和通信计数 |
| `generate_wall_s` | 本轮各 rank 的 generate 墙钟时间最大值 |
| `dit_forward_cuda_sum_s` | 各 rank 的本地 forward 区间合计，再取最大值；不是逐次取最大后求和 |
| `peak_allocated_bytes` / `peak_reserved_bytes` | 各卡完整 generate 峰值及最大每卡峰值，包含 rank 0 VAE |
| `video_per_s` / `generated_frames_per_s` | 每秒视频数和生成帧数；两个 GPU 合作生成一条视频，分子不乘 2 |
| `gpu_seconds_per_video` | `2×generate_wall_s`，GPU 资源占用近似量 |
| `summary` / `summary_by_rank` | 仅正式 runs 的中位数、最小值、最大值和均值；三次样本不报告 p95 |
| `pipeline_load_wall_s` | 对齐开始后的各 rank 组件加载时间最大值，独立于生成时间 |
| `environments` / `gpu_topology` | 每 rank GPU/软件/精度、官方及项目源码哈希、CPU 线程与双卡拓扑 |

通信统计只覆盖 DiT 内实际完成的 SUM AllReduce：Row Linear 记录完整输出部分和的张量字节数，Q/K RMSNorm 记录 FP32 `[...,1]` 统计张量字节数。通过模块 hooks 观测，不替换 collective，不同步每次通信。字节数是每 rank 逻辑 reduction tensor 大小，不是网络传输总量、NCCL bus bandwidth 或通信耗时；文本/noise 广播、管线 barrier、benchmark gather 等不计入该 DiT 统计。正式速度测量不启用 profiler；NCCL 时间线和重叠分析后续独立采集。新增形状/dtype 计数 hooks 的 CPU 开销也属于本入口的观测开销。

可选传入 `--baseline-metrics /path/to/single_gpu/metrics.json`，使用已有单卡报告计算管线和 DiT 加速比、对应效率以及 allocated 显存降低比例。对比要求 baseline 已完成，workload、正负提示词、solver、精度、offload、GPU 型号、CPU intra-op 线程及关键软件/源码元数据一致；不匹配时拒绝输出加速比，已测数据保留。metadata 相同不能证明两次租赁的 CPU 资源、拓扑和运行环境完全可比，仍需人工审阅环境记录。当前 rank 0 独占 T5/VAE 的资源组织也明确写入报告，不能把全部端到端收益都归因于 DiT TP。

这个入口只验证输出形状、有限值和 forward 调用数，不证明 TP 与官方模型数值一致或视频质量正确，也不验证逐步跨 rank latent 一致性。完整 denoising 区间尚未独立计时，不能把 DiT CUDA 区间合计改称 denoising latency。TP2 benchmark 源码仅经人工审阅，尚未在云端执行。

### TP size 的当前边界

生成入口使用显式 TP size，没有将底层分片写死为 2，但要求 WORLD size 相等，并要求 Linear 维度、hidden dimension 与完整 attention heads 可整除。T2V-1.3B 的 hidden=1536、FFN=8960、heads=12；TP8 因 `12 % 8 != 0` 无法按现有完整 head 规则切分，启动 8 个进程也会在 Attention 构造时失败。当前模型满足这些整除条件的 TP sizes 为 1、2、4；这只是结构条件，TP4 未验证，不能宣称已支持运行或扩展性能。当前 benchmark 只面向 TP2。

## 已有 TP 接口

从工程根目录使用以下模块路径：

```python
from parallel.tp_linear import ColumnParallelLinear, RowParallelLinear
from parallel.tp_weight_loader import (
    load_linear_weight,
    load_rmsnorm_weight,
    load_attention_weights,
    load_replicated_weights,
)
from models.wan.tp_wan import (
    create_wan_tp_model,
    parallelize_wan_model,
    load_wan_tp_weights,
)
```

模型适配层的替换及加载入口统一接收 `WanModel`，因为它们依赖 Wan 的 `blocks` 及子模块结构；替换函数原地修改并返回同一个模型，加载函数原地填充参数并返回 `None`。底层 Linear、RMSNorm 和 Attention 则继承 `nn.Module`，各自的 loader 接收具体 TP 类型。`TPWanT2VCrossAttention` 继承 `TPWanSelfAttention`，共用 Attention 加载接口。

调用方负责初始化进程组及维护其生命周期。TP Linear 构造器根据维度、进程组、device 和 dtype 创建未初始化的分片参数；首次 forward 前，统一 loader 读取组内 rank 对应的 checkpoint 分片与完整复制参数并原地填充。

Column 切分输出维及 bias，输出保持局部分片。Row 消费已分片的输入，归约输出部分和后加一次完整 bias。所有 rank 需要一致的输入、checkpoint 和通信调用顺序；当前接口面向推理。

## 环境与状态

### 分布式运行接口

运行入口显式组织两层资源：分布式环境管理 WORLD，模型并行状态管理 TP。接口为：

```text
runtime.distributed
  init_distributed_environment(*, timeout: timedelta) → device
  destroy_distributed_environment() → None

runtime.parallel_state
  initialize_model_parallel(tensor_model_parallel_size, *, timeout: timedelta) → None
  get_tensor_model_parallel_group() → ProcessGroup
  destroy_model_parallel() → None
```

生命周期为：

```text
torchrun 启动进程，设置 rendezvous 和 rank 环境变量
  → init_distributed_environment(timeout=...)
  → 根据 LOCAL_RANK 绑定当前 CUDA device
  → env:// 初始化 WORLD，传入 device_id 和 timeout，返回 device
  → initialize_model_parallel(tensor_model_parallel_size=2, timeout=...)
  → 为所有 WORLD 成员显式创建独立 NCCL TP 组
  → get_tensor_model_parallel_group() 得到模型使用的组
  → 调用方完成模型构造、加载及所有推理通信
  → 运行入口在 finally 中先 destroy_model_parallel()
  → 再 destroy_distributed_environment()
```

当前只支持一个纯 TP 推理组，模型并行初始化要求 `WORLD_SIZE == tensor_model_parallel_size`，不实现 DP/PP 或多个独立 TP 副本。TP2 的数字 2 由运行入口配置，不写死在环境初始化中。TP 与 WORLD 当前成员相同，但分别创建和销毁；这是职责和生命周期划分，不代表性能会优于直接使用 WORLD。

模型构造时传入初始化返回的 `device` 和 getter 返回的 TP `process_group`。模型并行状态只保存 TP 组对象，rank/size 使用 `torch.distributed.get_rank(process_group)` 和 `get_world_size(process_group)` 查询，不重复保存副本。`LOCAL_RANK` 用于 GPU 选择，不能用全局 rank 代替；GPU 编号相对于当前进程可见的 CUDA 设备。

运行入口拥有默认组与 TP 组，不借用外部已初始化的默认组。必要检查限于重复初始化、初始化顺序、纯 TP 成员范围和 getter 未初始化调用。清理函数在对应资源尚未创建时不执行销毁，供初始化中途失败后的 finally 路径使用；已创建的 TP 组必须先于 WORLD 销毁。所有 rank 需要一致的初始化、通信和销毁顺序。接口不增加退出 barrier、自动重试或 CPU/Gloo fallback，不提供跨 rank 故障恢复；timeout 由调用方显式传入。输入、seed 与 scheduler 一致性由后续生成管线负责。

职责划分参考 vLLM `v0.10.2` 的分布式环境与模型并行生命周期，以及 Megatron `core_v0.13.0` 的显式 TP 组创建。GPU 绑定沿用固定版本官方 Wan 的路径，NCCL 初始化使用云端 PyTorch 2.8 的 `device_id` 接口。当前裁剪了 DP/PP/SP 组、Gloo 控制组和自定义 communicator。生成入口已编写，尚未执行启动验证。

本机只负责编辑与人工审阅。依赖安装、模型加载、执行和验证在云端 `/root/autodl-tmp/wan_xdit` 的 `.venv` 中进行，已记录 Python 版本为 3.12；上述独立仓库布局从其 `wan_parallel` 子目录启动，官方源码路径为 `../Wan2.1`。

`pyproject.toml` 声明当前 TP 模块和管线的直接依赖与包发现范围，包括用于官方管线配置类型的 `easydict` 和采样进度显示的 `tqdm`。官方 Wan 环境按外层云端准备记录维护。本轮没有安装依赖、构建或安装项目包，也没有执行语法、import、lint、CPU、GPU 或性能验证。

已移除没有测试函数的 test 占位文件和没有测量逻辑的 benchmark 占位文件；当前没有 TP 数值正确性对比脚本，后续用户明确要求云端验证时再编写。meta 构造链条、Attention/FFN TP、完整 DiT 参数加载、NCCL runtime 接口、TP 管线初始化和采样、生成 CLI 及 TP2 benchmark 已经编写，仅经源码人工审阅，尚未执行云端验证。单卡基线入口已在迁移前取得云端结果；迁移与路径调整未执行验证。SP、SP 性能测量及 NCCL profiler 时间线采集尚未实现。
