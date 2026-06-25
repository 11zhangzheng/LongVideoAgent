# LongVideoAgent 复现指南

本文档基于当前仓库 `main` 分支整理，用于在本地复现 LongVideoAgent 的数据准备、评测和 GRPO 训练流程。

## 0. 推荐复现顺序（快速复现运行）

最小闭环：

```bash
conda create -n lvagent python=3.11
conda activate lvagent
cd /home/zhangzheng/projects/LVAgent
pip install -e . --no-deps
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install wandb

bash scripts/download_and_prepare_longtvqa.sh

# 先跑 API 评测，确认数据和 API 可用
bash scripts/eval_unified_api.sh

# 再下载官方 checkpoint，跑本地 Master Agent 评测
bash scripts/eval_unified_local.sh

# 最后准备 parquet 并启动 GRPO 训练
python src/dataset/build_grounding_cache.py ...
python src/dataset/convert_tvqa_json_to_grpo_parquet.py ...
bash scripts/quickstart_qwen_2_5_3B_grpo.sh
```


## 1. 硬件与外部依赖

建议环境：

- Linux + NVIDIA GPU
- CUDA 12.4 作为官方可运行组合参考
- Python 3.11
- PyTorch 2.5.1 cu124
- vLLM 0.7.3
- transformers 4.57.6

评测和训练通常还需要 OpenAI-compatible API 服务：

- Grounding Agent API
- Vision Agent API
- 如果使用纯 API 评测，还需要 Main Agent API

脚本中的默认模型名和地址包括：

```bash
grok-4-fast-reasoning
grok-4-fast-non-reasoning
gpt-4o
https://api2.aigcbest.top/v1
https://dashscope.aliyuncs.com/compatible-mode/v1
```

这些不是必须固定值，可替换为你自己的兼容接口和模型。

## 2. 创建环境

官方推荐：

```bash
conda create -n lvagent python=3.11
conda activate lvagent
cd /home/zhangzheng/projects/LVAgent

pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

如果依赖冲突，使用无依赖安装方式：

```bash
pip install -e . --no-deps
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install wandb
```

如需 Hugging Face 数据下载工具：

```bash
pip install -U "huggingface_hub[cli]"
```

## 3. 准备 API Key

不同脚本支持直接传参，也支持环境变量。建议先设置环境变量：

```bash
export qdd_api="你的 grounding/main API key"
export aliyun_api="你的 vision API key"
```

训练脚本还读取这些变量或脚本内变量：

```bash
export AGENT_SHARED_API_KEY=""
export GROUNDING_API_KEY=""
export VISION_API_KEY=""
export MAIN_API_KEY=""
```

如果 `AGENT_SHARED_API_KEY` 非空，训练里的 vision 配置会把它作为共享 key 传入。

## 4. 下载与整理数据

一键脚本：

```bash
bash scripts/download_and_prepare_longtvqa.sh
```

默认输出位置在项目上一层：

```bash
/home/zhangzheng/projects/Tvqa_data
```

默认生成或下载：

```text
Tvqa_data/hf_datasets/LongTVQA
Tvqa_data/hf_datasets/LongTVQA_plus
Tvqa_data/frames
Tvqa_data/LongTVQA_val_normalized.jsonl
Tvqa_data/LongTVQA_plus_val_normalized.json
Tvqa_data/clip_bbox_mapping.json
```

脚本默认只下载并解压 `bbt` 的帧。可通过环境变量切换剧集：

```bash
FRAMES_SHOW=house bash scripts/download_and_prepare_longtvqa.sh
```

支持的值见脚本：

```text
bbt, castle, friends, grey, house, met
```

如需自定义数据目录：

```bash
DATA_DIR=/path/to/Tvqa_data OUTPUT_DIR=/path/to/Tvqa_data bash scripts/download_and_prepare_longtvqa.sh
```

## 5. 快速评测：API 模式

API 模式下 Main Agent、Grounding Agent、Vision Agent 都通过 API 调用，不需要本地加载主模型。

先编辑：

```bash
scripts/eval_unified_api.sh
```

至少修改：

```bash
DATASET="tvqa_plus"       # tvqa 或 tvqa_plus
GROUNDING_API_KEY="..."
VISION_API_KEY="..."
MAIN_API_KEY="..."
```

确认默认数据路径是否匹配你的实际数据。当前脚本里 `tvqa_plus` 默认使用：

```text
../Tvqa_data/tvqa_plus_val.json
../Tvqa_data/all_episodes_subtitles_by_clips.json
../Tvqa_data/bbt_frames
../Tvqa_data/clip_bbox_mapping.json
```

如果你使用的是一键脚本生成的 normalized 文件，建议在脚本中显式改为实际存在的文件，例如：

```bash
QUESTIONS_PATH="../Tvqa_data/LongTVQA_plus_val_normalized.json"
BBOX_JSON_PATH="../Tvqa_data/clip_bbox_mapping.json"
BASE_FRAME_DIR="../Tvqa_data/frames/bbt_frames"
```

运行：

```bash
bash scripts/eval_unified_api.sh
```

也可以直接调用 Python：

```bash
python src/evaluation/lvagent/evaluate_api_unified.py \
  --dataset tvqa_plus \
  --checkpoint_step api \
  --max_turn 5 \
  --threads 30 \
  --questions-path ../Tvqa_data/LongTVQA_plus_val_normalized.json \
  --subs-path ../Tvqa_data/all_episodes_subtitles_by_clips.json \
  --base-frame-dir ../Tvqa_data/frames/bbt_frames \
  --bbox-json-path ../Tvqa_data/clip_bbox_mapping.json \
  --output-filename ./results/tvqa_plus_api_summary.json \
  --detailed-output-filename ./results/tvqa_plus_api_detail.json \
  --grounding-model grok-4-fast-reasoning \
  --vision-model gpt-4o \
  --main-model grok-4-fast-reasoning \
  --grounding-base-url https://api2.aigcbest.top/v1 \
  --vision-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --main-base-url https://api2.aigcbest.top/v1 \
  --grounding-api-key "$qdd_api" \
  --vision-api-key "$aliyun_api" \
  --main-api-key "$qdd_api"
```

## 6. 本地主模型评测

本地评测使用 vLLM 加载 Master Agent，Grounding/Vision 仍走 API。

可使用官方发布权重，例如：

```text
longvideoagent/longvideoagent-qwen2.5-3b
longvideoagent/longvideoagent-qwen2.5-7b
```

如需下载：

```bash
huggingface-cli download longvideoagent/longvideoagent-qwen2.5-3b \
  --local-dir /path/to/models/longvideoagent-qwen2.5-3b
```

编辑：

```bash
scripts/eval_unified_local.sh
```

至少修改：

```bash
DATASET="tvqa_plus"
LLM_PATH="/path/to/models/longvideoagent-qwen2.5-3b"
GROUNDING_API_KEY="..."
VISION_API_KEY="..."
```

根据你的数据实际位置修改：

```bash
QUESTIONS_PATH="../Tvqa_data/LongTVQA_plus_val_normalized.json"
BASE_FRAME_DIR="../Tvqa_data/frames/bbt_frames"
BBOX_JSON_PATH="../Tvqa_data/clip_bbox_mapping.json"
```

运行：

```bash
bash scripts/eval_unified_local.sh
```

也可直接调用：

```bash
python src/evaluation/lvagent/evaluate_local_unified.py \
  --dataset tvqa_plus \
  --llm-path /path/to/models/longvideoagent-qwen2.5-3b \
  --max_turn 5 \
  --gpu_memory_utilization 0.4 \
  --questions-path ../Tvqa_data/LongTVQA_plus_val_normalized.json \
  --subs-path ../Tvqa_data/all_episodes_subtitles_by_clips.json \
  --base-frame-dir ../Tvqa_data/frames/bbt_frames \
  --bbox-json-path ../Tvqa_data/clip_bbox_mapping.json \
  --output-filename ./results/tvqa_plus_local_summary.json \
  --detailed-output-filename ./results/tvqa_plus_local_detail.json \
  --grounding-model grok-4-fast-reasoning \
  --vision-model gpt-4o \
  --grounding-base-url https://api2.aigcbest.top/v1 \
  --vision-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --grounding-api-key "$qdd_api" \
  --vision-api-key "$aliyun_api"
```

### 6.1 分阶段稳定性评测

当前服务器上的推荐本地模型与数据路径已经写入：

```bash
scripts/eval_unified_local.sh
```

单阶段 smoke test：

```bash
EVAL_LIMIT=10 bash scripts/eval_unified_local.sh
```

按 `10 -> 20 -> 50 -> full` 自动分阶段评测：

```bash
bash scripts/run_staged_local_eval.sh
```

如需从更小阶段开始：

```bash
STAGES="1 10 20 50 full" bash scripts/run_staged_local_eval.sh
```

每个阶段会生成独立文件：

```text
results/stability_local/local_tvqa_plus_<stage>_<policy>_summary.json
results/stability_local/local_tvqa_plus_<stage>_<policy>_detail.json
results/stability_local/local_tvqa_plus_<stage>_<policy>.log
results/stability_local/local_tvqa_plus_<stage>_<policy>_checkpoint.jsonl
results/stability_local/report_<policy>.md
```

`report.md` 会汇总：

```text
Accuracy
Completion rate
Average turns
Vision calls / question
Frames / question
Clips / question
Grounding calls / question
Time / question
Failed cases
```

如果 V100 被调度到可见 GPU，需要使用半精度：

```bash
VLLM_DTYPE=half bash scripts/run_staged_local_eval.sh
```

如果确认使用 A800，可指定卡号：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_staged_local_eval.sh
```

只运行或恢复 full validation：

```bash
CUDA_VISIBLE_DEVICES=3 VISION_POLICY=dynamic STAGES="full" \
  bash scripts/run_staged_local_eval.sh
```

full 阶段的逐题断点文件为：

```text
results/stability_local/local_tvqa_plus_full_dynamic_checkpoint.jsonl
```

每完成一题都会立即追加到 checkpoint。命令中断后，重新执行同一条命令即可从下一题继续。若要放弃断点并从头开始，可使用新的 `STABILITY_DIR`，或将 `RESUME_EVAL=0` 直接运行 `scripts/eval_unified_local.sh`。

### 6.2 动态 Vision 策略

本地评测支持两种策略：

```text
VISION_POLICY=fixed    原始固定策略，当前 clip 固定采样约 15 帧
VISION_POLICY=dynamic  根据问题类型、字幕证据和历史调用动态决策
```

动态策略默认行为：

```text
dialogue:         4-6 帧，最多 1 次 Vision，1 个 clip
visual_detail:    8-16 帧，最多 2 次 Vision
temporal_action: 14-20 帧，最多 2 次 Vision
causal_emotion:   8-16 帧，最多 2 次 Vision
```

第二次 Vision 搜索可根据 `before/after` 等时序词扩展到相邻 clip。可使用以下环境变量调整预算：

```bash
VISION_MIN_FRAMES=4
VISION_MAX_FRAMES=20
VISION_MAX_CLIPS=2
VISION_MAX_CALLS=2
```

先进行 10/20/50 样本 A/B 测试：

```bash
CUDA_VISIBLE_DEVICES=3 VISION_POLICY=fixed STAGES="10 20 50" \
  bash scripts/run_staged_local_eval.sh

CUDA_VISIBLE_DEVICES=3 VISION_POLICY=dynamic STAGES="10 20 50" \
  bash scripts/run_staged_local_eval.sh
```

对比报告：

```text
results/stability_local/report_fixed.md
results/stability_local/report_dynamic.md
```

不同策略和预算具有独立 checkpoint。恢复时若策略配置发生变化，程序会拒绝混用旧 checkpoint，避免污染实验结果。

## 7. 训练数据准备

### 7.1 构建离线 grounding cache

官方强烈建议先构建 grounding cache，否则训练数据转换时会随机选初始 clip，效果会明显变差。

```bash
python src/dataset/build_grounding_cache.py \
  --dataset tvqa_plus \
  --questions-path /path/to/train.json \
  --subs-path /path/to/all_episodes_subtitles_by_clips.json \
  --grounding-model grok-4-fast-reasoning \
  --grounding-base-url https://api2.aigcbest.top/v1 \
  --grounding-api-key "$qdd_api" \
  --output-dir /path/to/cache_dir \
  --threads 8
```

小样本 smoke test：

```bash
python src/dataset/build_grounding_cache.py \
  --dataset tvqa_plus \
  --questions-path /path/to/train.json \
  --subs-path /path/to/all_episodes_subtitles_by_clips.json \
  --grounding-model grok-4-fast-reasoning \
  --grounding-base-url https://api2.aigcbest.top/v1 \
  --grounding-api-key "$qdd_api" \
  --output-dir /path/to/cache_dir \
  --threads 4 \
  --max-samples 20 \
  --overwrite
```

### 7.2 转换为 GRPO parquet

训练脚本默认读取：

```text
./data/train.parquet
./data/val.parquet
```

转换命令：

```bash
python src/dataset/convert_tvqa_json_to_grpo_parquet.py \
  --questions-path /path/to/LongTVQA_or_LongTVQA_plus_questions.jsonl_or_json \
  --grounding-cache-json /path/to/cache_dir/grounding_cache_tvqa_plus_grok-4-fast-reasoning.json \
  --subtitles-dir /path/to/subtitles_dir \
  --output-dir ./data \
  --seed 42
```

`--subtitles-dir` 目录必须包含：

```text
LongTVQA_plus_subtitle_clip_level.json
LongTVQA_plus_subtitle_episode_level.json
```

如果只想先跑小规模流程：

```bash
python src/dataset/convert_tvqa_json_to_grpo_parquet.py \
  --questions-path /path/to/questions.json \
  --grounding-cache-json /path/to/cache.json \
  --subtitles-dir /path/to/subtitles_dir \
  --output-dir ./data \
  --subset-size 100 \
  --seed 42
```

## 8. GRPO 快速训练

官方快速脚本：

```bash
bash scripts/quickstart_qwen_2_5_3B_grpo.sh
```

运行前必须检查并修改脚本顶部配置：

```bash
export RAY_TMPDIR=/home/rliuay/runtao/proj_videoqa
export BASE_MODEL='../Qwen2.5-3B-Instruct'
export TRAIN_DATA_DIR='./data'
export TEST_DATA_DIR='./data'
VISION_BASE_FRAME_DIR="../Tvqa_data/frames/bbt_frames"
VISION_BBOX_JSON_PATH="../Tvqa_data/clip_bbox_mapping.json"
```

建议改成你的本机路径，例如：

```bash
export RAY_TMPDIR=/tmp/ray_lvagent
export BASE_MODEL='/path/to/Qwen2.5-3B-Instruct'
export TRAIN_DATA_DIR='/home/zhangzheng/projects/LVAgent/data'
export TEST_DATA_DIR='/home/zhangzheng/projects/LVAgent/data'
VISION_BASE_FRAME_DIR="/home/zhangzheng/projects/Tvqa_data/frames/bbt_frames"
VISION_BBOX_JSON_PATH="/home/zhangzheng/projects/Tvqa_data/clip_bbox_mapping.json"
```

GPU 设置也需要核对：

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE=1
```

如果只用单卡，建议：

```bash
export CUDA_VISIBLE_DEVICES=0
N_GPUS_PER_NODE=1
```

如果用多卡，需要同步调整：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=4
```

训练输出：

```text
../longvideoagent_trainYYYYmmdd_HHMMSS
verl_checkpoints/<experiment_name>
<experiment_name>.log
```

注意：脚本末尾打印的 `verl_checkpoints/${EXPERIMENT_NAME}` 与实际 `trainer.default_local_dir=../$EXPERIMENT_NAME` 可能不一致，复现时以实际生成目录为准。

## 9. LoRA 合并

训练使用 LoRA。训练完成后可将 adapter 合并到基础模型：

```bash
bash scripts/merge_lora_adapter.sh \
  --base_model /path/to/Qwen2.5-3B-Instruct \
  --adapter_path /path/to/lora_adapter \
  --output_dir /path/to/merged_hf_model \
  --dtype bf16 \
  --device_map auto \
  --trust_remote_code
```

如果 adapter 是单个 safetensors 文件：

```bash
bash scripts/merge_lora_adapter.sh \
  --base_model /path/to/Qwen2.5-3B-Instruct \
  --adapter_path /path/to/adapter.safetensors \
  --adapter_config /path/to/adapter_config.json \
  --output_dir /path/to/merged_hf_model
```

合并后的目录可作为 `evaluate_local_unified.py --llm-path` 使用。

## 10. 常见问题

### flash-attn 安装失败

优先确认 PyTorch、CUDA、Python 版本匹配。官方参考组合是：

```text
cuda12.4, torch==2.5.1 cu124, vllm==0.7.3, transformers==4.57.6
```

### 下载 Hugging Face 数据失败

确认安装了 CLI：

```bash
hf --help
```

或：

```bash
huggingface-cli --help
```

国内网络环境可能需要代理或镜像。

### 评测脚本提示文件不存在

优先核对以下路径：

```bash
QUESTIONS_PATH
SUBS_PATH
BASE_FRAME_DIR
BBOX_JSON_PATH
GROUNDING_CACHE_JSON_PATH
```

仓库脚本的默认路径和一键下载脚本生成的 normalized 文件名并不完全一致，需要按实际文件修改。

### API Key 未生效

统一评测脚本优先使用脚本内变量。如果脚本里 `GROUNDING_API_KEY=""`，即使你设置了环境变量，脚本也可能直接报错。最稳妥做法是在脚本中填入，或绕过 shell 脚本直接调用 Python 并传 `--grounding-api-key` 等参数。

### 训练启动后显存不足

先降低：

```bash
TRAIN_BSZ
VAL_BSZ
ROLLOUT_N
VLLM_GPU_MEMORY_UTILIZATION
MAX_PROMPT_LEN
MAX_RESP_LEN
```

并确认 `N_GPUS_PER_NODE` 与 `CUDA_VISIBLE_DEVICES` 数量一致。
