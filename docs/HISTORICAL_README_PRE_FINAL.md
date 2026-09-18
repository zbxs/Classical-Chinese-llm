# 古文小模型：CPT → SFT → RM / DPO / GRPO 全流程实验

本项目以 **Qwen2.5-0.5B Base** 为主模型，在 Windows 11、RTX 4060 Laptop 8GB 上搭建一条可复现的古文增量训练研究流水线。项目覆盖数据采集与许可记录、清洗去重、按 token 精确混配、增量预训练（CPT）、100K 指令数据构造、奖励模型、DPO、GRPO、四类古文评测和通用能力/灾难性遗忘检查。

当前已经完成并真实运行的是：隔离环境、资源复用、公开数据流式采样、真实数据清洗与混配、五个训练阶段各 1 step 的 GPU 冒烟测试、教师合成审计测试、四类生成评分和困惑度冒烟测试。**完整 40B-token 预训练、100K SFT、20K 强化学习数据训练及 4×100 正式评测尚未运行，因此本项目不声称已有最终模型优劣结论。** 40B 方案保留为集群配置；本地建议先做 50M–500M token 的研究规模消融。

## 1. 为什么采用这套设计

目标不是只得到一个 LoRA 文件，而是能回答以下研究问题：

1. 古文数据占 5%、20%、50% 时，领域能力提升与通用能力遗忘如何变化？
2. 0.5B 与 1.5B 模型在 50M、200M、500M token 下的数据利用率是否不同？
3. 相同 SFT 起点和相同 20K 问题下，DPO 与使用奖励模型的 GRPO 哪种更稳定？
4. 训练 loss 的改善是否真正转化为古译今、今译古、鉴赏和创作能力？

本地 8GB 显存采用 NF4 QLoRA、BF16、gradient checkpointing 和梯度累积。40B token 对一张 4060 的时间与存储成本并不现实，见 `configs/profiles/full_40b.yaml`；该配置明确 `run_locally: false`，防止误执行。

## 2. 目录说明

```text
project1/
├─ configs/
│  ├─ data/                 # 数据源、token 配比与消融设计
│  ├─ evaluation/           # 领域 4×100 与通用困惑度配置
│  ├─ experiments/          # 本地正式 CPT/SFT/RM/DPO/GRPO 配置
│  ├─ profiles/             # 40B 集群方案（本地禁用）
│  └─ smoke/                # 各训练阶段 1-step 配置
├─ data/
│  ├─ examples/             # 可提交的小型单元/冒烟样例
│  ├─ raw/                  # 下载原始数据和 acquisition.json（忽略提交）
│  ├─ cleaned/              # 清洗、去重、稳定切分与 audit.json
│  ├─ sft/                  # 公开对齐 SFT 和教师候选审计
│  ├─ preferences/          # chosen/rejected 偏好对
│  ├─ final/                # 复核后才允许进入的正式训练/评测数据
│  └─ manifests/            # 复用资源哈希和来源清单
├─ docs/
│  ├─ DATA_PIPELINE.md      # 数据如何生成、筛选、拆分和复核
│  └─ EXPERIMENT_PROTOCOL.md# 选 checkpoint、算法公平比较、遗忘判定
├─ outputs/                 # checkpoint、best adapter、指标（忽略提交）
├─ reports/
│  ├─ environment.json      # 环境与 GPU 实测
│  ├─ schemas/              # 数据集真实 schema 抽查
│  ├─ SMOKE_RESULTS.md      # 已运行结果与适用边界
│  └─ generated/            # 自动汇总的 run metadata 与评测结果
├─ requirements/            # 分层依赖与当前环境锁文件
├─ scripts/                 # 一键数据、训练、评测脚本
├─ src/classical_llm/
│  ├─ data/                 # acquire/prepare/dedup/mix/SFT/preference/teacher
│  ├─ training/             # CPT/SFT/RM/DPO/GRPO
│  ├─ evaluation/           # 生成、自动评分、困惑度
│  └─ cli.py                # 统一命令入口
├─ tests/                   # 数据泄漏、配额、去重、格式等测试
├─ prompt.txt               # 原始需求
└─ pyproject.toml
```

## 3. 隔离环境与已复用资源

所有 Python 包安装在项目内的 `.venv`，pip、Hugging Face、Torch、Triton 缓存分别写入项目的 `.cache/*`；`PYTHONNOUSERSITE=1` 阻止用户级 site-packages 混入。不会全局安装依赖。

```powershell
# 新机器：创建项目内环境并安装依赖
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -Dev

# 每个新终端先激活；同时设置所有项目本地缓存变量
.\scripts\activate.ps1

# 确认解释器、CUDA、显存和关键库
classical-llm doctor
```

本机实测：Python 3.12.10、PyTorch 2.6.0+cu124、CUDA 可用、BF16 可用、RTX 4060 Laptop 8GB。完整记录见 `reports/environment.json`。

为节省下载，项目只读复用了旧“科研规划”项目中的 PyTorch CUDA wheel 和 Qwen2.5-0.5B-Instruct；路径及 SHA-256 在 `data/manifests/reused_resources.json`。主训练所需的 Qwen2.5-0.5B **Base** 单独下载到当前项目缓存，因为 Instruct 模型不适合作为 CPT 的干净起点。资源复核命令：

```powershell
classical-llm audit-resources --config configs/local_resources.yaml
```

旧资源不会被修改或删除。模型/数据缓存体积大，已由 `.gitignore` 排除。

## 4. 数据获取、许可与版本固定

来源定义集中在 `configs/data/sources.yaml`：

| 用途 | 数据集 | 子集 | 许可 | 当前策略 |
|---|---|---|---|---|
| 古籍原文 | [gujilab/chinese-classical-corpus](https://huggingface.co/datasets/gujilab/chinese-classical-corpus) | corpus | CC0-1.0 | CPT、教师任务的来源文本 |
| 双向翻译 | 同上 | translate | CC0-1.0 | SFT、偏好参考答案 |
| 古文标点 | 同上 | punctuate | CC0-1.0 | CPT 辅助，不计入四类最终任务 |
| 通用中文 | [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | cmn_Hani | ODC-By-1.0 | 防止中文通用能力遗忘 |
| 通用英文 | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | sample-10BT | ODC-By-1.0 | 保留部分跨语言通用能力 |
| 中文高质候选 | [CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ) | — | 用户协议 | 默认关闭，审阅并授权后才启用 |

gujilab 固定到 commit `1979f15b4b749b3601fd9ee3e4111a725c079439`；流式网页语料把 dataset/config 和采集清单同时保存。公开许可不等于所有下游用途没有义务，发布衍生模型前仍应复核 ODC-By 署名和基础模型许可证。

已执行的真实小样本采集：五个启用来源各 100 条，见 `data/raw/huggingface_sample/acquisition.json`。重跑：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/01_data_sample.ps1 -MaxDocuments 100
```

把 `MaxDocuments` 增大即可扩展采样；不要省略存储预算检查后直接拉取 40B token。采集器逐源容错，一个来源失败会在清单中留下错误，不会把空文件当成功。

## 5. 清洗、去重与防止数据泄漏

处理顺序固定如下：

1. Unicode NFC 和空白规范化；保留繁体、简体、异体字，不做不可逆“统一”。
2. 质量规则：过短/过长、乱码替代符、12 个以上相同字符、网页模板、重复行、中文比例。
3. SHA-256 精确去重。
4. 字符三元组 64 位 SimHash 近重复去重，汉明距离阈值 3。
5. 先按作品 `work` 分组；缺失时依次使用 source_group、source_id、URL、id。组键经固定 seed 哈希后切分为 90%/5%/5%。同一作品的不同章节不会跨 train/validation/test。

每个拒绝原因均进入 `audit.json`。本次 500 条真实采样保留 491 条：6 条近重复、3 条异常连续字符；train/validation/test 为 464/13/14。不能只看总数，还应抽查被拒绝和超长文本。

## 6. CPT 数据如何精确组成

正式目标是古文相关 20%、通用 80%。本地默认把通用部分细分为中文 60%、英文 15%、理工技术 5%。混配不是按文件数或字符数：先用 Qwen tokenizer 统计 token，长篇按不超过 2,048 token 切块，最后一块按剩余额度截断；CPT 装块时在不同文档之间插入 EOS。

100K-token 真实样本试跑因尚未引入独立理工源，采用古文 20%、通用中文 65%、通用英文 15%，清单精确得到 20,000 / 65,000 / 15,000 token。每类文档默认最多复用两轮；语料不足会报错，不会无限重复凑预算。

本地研究矩阵：

- 数据配比：古文 5%、20%、50%，其他为通用数据；
- token 规模：50M、200M、500M；
- 模型规模：Qwen2.5-0.5B，资源允许时加入 1.5B；
- 相同 held-out validation，比较 eval loss、领域分、通用困惑度、吞吐和显存。

40B 版本需要分布式训练、流式可恢复数据、远端对象存储和分片优化器状态。它不是本地一键脚本的默认路径。

## 7. CPT 训练与 checkpoint 选择

```powershell
classical-llm train cpt --config configs/experiments/cpt_local.yaml
```

CPT 使用 Base 模型和 causal language modeling。训练文本先 token 化并形成 EOS 分隔的块，避免 Windows 环境没有 FlashAttention 时，TRL 的 padding-free 打包让不同文档发生错误的跨文档注意力。默认每 250 step 评估和保存，`load_best_model_at_end` 按 validation `eval_loss` 选择最优点，最终把适配器和 tokenizer 保存到 `best/`。训练 loss 不作为 checkpoint 最终依据。

若要断点续训，在配置中设置 `resume_from_checkpoint`。每次运行都会写 `run_metadata.json`，其中包含完整配置、输入模型、Python/Torch/CUDA/GPU 和实际指标。

## 8. 100K SFT 数据具体如何生成

目标分配不是 token 数，而是 **100,000 条训练样本**：

| 任务 | 目标训练条数 | 主要生成方式 |
|---|---:|---|
| 古译今 + 今译古 | 50,000 | 公开 source/reference 对，双向分别统计 |
| 翻译与鉴赏 | 25,000 | held-in 古文切片 + 教师生成 + 文本依据规则 + 人工复核 |
| 仿写与语境运用 | 25,000 | 结构化约束任务 + 教师候选 + 防照抄规则 + 人工复核 |

直接标注数据经空值、长度比、重复和来源检查后转换为 system/user/assistant 消息。`--quotas` 对每个任务要求精确训练条数；不足会报错，绝不复制已有样本。validation/test 是训练配额之外的数据。

```powershell
classical-llm build-sft data/cleaned/.../train.jsonl data/cleaned/.../validation.jsonl `
    --output data/final/sft `
    --quotas '{"old_to_modern":25000,"modern_to_old":25000,"appreciation":25000,"creation":25000}'
```

鉴赏/创作候选使用独立教师阶段：只接受 `category=classical` 且中文比例至少 65% 的来源；长文按句切到 48–420 字。鉴赏答案必须出现连续四字原文证据；创作答案若复制来源连续八字即拒绝。所有输出同时写入审计文件，带来源、教师路径、参数、规则结果，并始终标记 `requires_human_review=true`。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/02_teacher_smoke.ps1 -Limit 2
```

本地复用的 0.5B Instruct 只适合验证程序。真实试跑 2 条只通过 1 条规则；更早的规则还曾让英文网页进入鉴赏生成并诱发幻觉，随后已修复来源门槛。这一实测说明正式 50K 合成数据必须使用更强教师并进行分层人工复核，不能把“规则通过”当“事实正确”。详细规范见 `docs/DATA_PIPELINE.md`。

SFT 训练：

```powershell
classical-llm train sft --config configs/experiments/sft_local.yaml
```

模型从 CPT adapter 继续训练，loss 只作用于 assistant completion，system/user prompt 不参与目标 loss。

## 9. 20K 偏好数据、奖励模型、DPO 与 GRPO

20K 指的是相同的 20,000 个问题。chosen 取复核后的参考答案；rejected 优先由当前 SFT 策略模型多次采样并人工/规则标错。如果暂时没有模型采样，构造器可生成带错误标签的困难负例：否定关系翻转、数字替换、遗漏末句、现代口吻侵入、无依据套话或截断。它们适合启动实验，但正式结论应提高真实模型负例比例。

```powershell
classical-llm build-preferences --input data/final/sft/train.jsonl `
    --output data/final/preferences --limit 20000

classical-llm train reward --config configs/experiments/reward_local.yaml
classical-llm train dpo    --config configs/experiments/dpo_local.yaml
classical-llm train grpo   --config configs/experiments/grpo_local.yaml
```

奖励模型从同一 Base 建立序列分类头，学习 chosen 分数高于 rejected。DPO 和 GRPO 都从同一个 SFT checkpoint 开始；DPO 直接优化偏好差，GRPO 每个提示生成多个回答，组合奖励模型分、格式/约束奖励和防照抄规则。GRPO 日志中的训练 reward 不是最终评测分。

完整本地链可用 `scripts/04_local_training.ps1` 依次运行；它要求 `data/final` 中的数据已经复核且数量满足目标。某阶段失败后脚本立即停止。

## 10. 四模型领域评测与通用能力

最终比较对象：Base、CPT、SFT、DPO、GRPO；另报告 GRPO 使用的 reward model。领域题库四类各 100：

1. 古文翻译为现代汉语；
2. 现代汉语翻译为文言文；
3. 古诗文翻译与鉴赏；
4. 诗词歌赋创作/语境运用。

`build-eval` 只接受 `split=test` 且带参考答案的候选，每类不足 100 会直接报错。题库应在任何训练开始前冻结，作品组必须与所有训练集隔离。

```powershell
classical-llm build-eval data/reviewed/eval_candidates.jsonl `
    --output data/final/evaluation/domain_400.jsonl --per-task 100

classical-llm generate --model MODEL_PATH `
    --dataset data/final/evaluation/domain_400.jsonl `
    --output outputs/evaluation/MODEL.jsonl
classical-llm score --input outputs/evaluation/MODEL.jsonl `
    --output outputs/evaluation/MODEL.scored.jsonl
```

自动指标只做辅助：翻译/鉴赏报告 chrF，创作报告行数和关键词硬约束，所有任务报告非空率和长度。主评分采用配置中的 1–5 分量表：忠实/完整/流畅、意义保持/文言风格/流畅、事实性/文本依据/深度、约束遵守/风格/连贯/原创。应匿名打乱模型身份，至少双人评审并报告一致性、均值、置信区间和失败案例。

```powershell
# generations 参数包含五个模型各自的生成文件；命令同时生成保密 key 文件
classical-llm blind-review `
    --generations '{"base":"outputs/evaluation/base.jsonl","sft":"outputs/evaluation/sft.jsonl"}' `
    --output outputs/evaluation/blinded_review.jsonl

# 评审者在 scores 字典填入 1–5 分后汇总；未填完的行单独计数，不进入均值
classical-llm summarize-review `
    --input outputs/evaluation/blinded_review.jsonl `
    --key outputs/evaluation/blinded_review.key.json `
    --output outputs/evaluation/human_summary.json
```

通用能力不能只问几个聊天问题。`perplexity` 对训练前冻结的通用中文、英文和古文验证文本计算 token 加权 NLL/困惑度：

```powershell
classical-llm perplexity --model MODEL_PATH `
    --dataset data/final/evaluation/general_zh.jsonl `
    --output outputs/evaluation/MODEL.general_zh.ppl.json
```

训练后通用困惑度相对 Base 明显上升、同时公开通用基准下降，是灾难性遗忘信号。领域提升必须与这项代价一起报告。四题和困惑度程序冒烟可运行 `scripts/05_evaluation_smoke.ps1`；其样本量极小，不能用于模型排序。

## 11. 已执行结果

五个训练阶段已在本机完成 1 step 串联：

| 阶段 | train loss | 时间（秒） | 验证内容 |
|---|---:|---:|---|
| CPT | 2.7064 | 1.8102 | 4-bit QLoRA、EOS 装块、eval/checkpoint |
| SFT | 3.5291 | 1.8863 | 从 CPT adapter 接续、completion-only loss |
| Reward | 0.6776 | 1.9407 | pairwise 分类头和保存/重载 |
| DPO | 0.6931 | 3.1566 | 从 SFT adapter 接续 |
| GRPO | 0.0000 | 6.5549 | 候选生成、RM 加载、组合奖励；单 step 数值无统计意义 |

运行全套冒烟：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/03_training_smoke.ps1
```

详细数字、教师失败案例和 4 题评测冒烟见 `reports/SMOKE_RESULTS.md`。输出指标由 `classical-llm report --output reports/generated/results.json` 自动汇总，不为未执行阶段填造结果。

## 12. 复现与验收顺序

建议按以下顺序检查项目：

```powershell
.\scripts\activate.ps1
classical-llm doctor
classical-llm audit-resources
python -m pip check
python -m ruff check src tests
python -m pytest
powershell -ExecutionPolicy Bypass -File scripts/01_data_sample.ps1
powershell -ExecutionPolicy Bypass -File scripts/03_training_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/05_evaluation_smoke.ps1
```

正式实验前的硬性门槛：

- `data/final` 中 100K SFT 和 20K 偏好问题数量准确且不靠复制；
- 合成行已完成人工复核并保留审计结果；
- 四类各 100 题和通用验证集已冻结，作品/来源不泄漏；
- CPT token 清单精确符合实验配比，许可清单无 unknown/gated 未授权项；
- 从 CPT 到 GRPO 的配置都指向预期 checkpoint；
- 报告同时呈现领域、通用、效率和失败案例，不用训练 loss 代替能力结论。

## 13. 当前限制与后续扩展

- 当前真实采集只是每来源 100 条的工程样本，不足以训练目标模型。
- gujilab 当前直接贡献的是古籍、翻译和标点；鉴赏/创作的 50K 数据必须经更强教师与人工复核补齐。
- 8GB 显存在短序列 QLoRA 上已验证，但正式 1,024-token、长时间训练仍需根据峰值显存调整梯度累积和生成长度。
- 40B token 不适合本机；若坚持该规模，应迁移到多 GPU 集群并实现分片 optimizer、可恢复流式数据和远端 checkpoint。
- 自动 chrF 对开放式鉴赏/创作不充分，最终结论必须依赖盲评量表。

研究协议细节见 `docs/EXPERIMENT_PROTOCOL.md`；任何结果表都应先确认它来自正式配置而不是 `configs/smoke`。

## 14. RTX 4090D 远端复现

远端项目目录、虚拟环境、模型缓存、数据和训练输出全部放在数据盘（示例：
`/root/autodl-tmp/Classical-Chinese-llm`）。Linux 虚拟环境通过
`--system-site-packages` 只复用租赁镜像已安装的 CUDA PyTorch；其余依赖全部安装在项目
`.venv`，不会写入全局 Conda 环境。Windows `.venv` 不迁移。

```bash
bash scripts/07_remote_setup.sh
source scripts/activate_remote.sh
python -m classical_llm.cli train cpt --config configs/remote/cpt_smoke_4090d.yaml
screen -dmS classical-llm bash -lc 'cd /root/autodl-tmp/Classical-Chinese-llm && bash scripts/08_production_pipeline_remote.sh'
```

4090D 配置保持有效批量为 16：CPT/SFT/Reward 用 `4 × 4`，DPO 和 GRPO 用
`1 × 16`。CPT/SFT 的 1,024-token smoke 实测表明 `8 × 2` 会因大词表 logits 达到
约 22.5 GiB 峰值并 OOM，因此正式配置保留梯度检查点并使用经验证的安全批量。训练阶段默认启用
Hugging Face、Transformers 和 Datasets 离线模式；依赖安装完成且项目内模型/数据存在后，
断网不影响训练。实时状态写入 `outputs/logs/production_state.json`，阶段日志写入
`outputs/logs/*.log`，编号 checkpoint 支持脚本重启后自动续训。

数据目录中的 JSONL 使用 Git LFS 发布；`.cache`、中间 checkpoint 和虚拟环境不进入 Git。
最终最佳 adapter、自动评测输出和完整运行元数据在实验完成后单独打包为 GitHub Release
资产，避免让普通 Git 历史无限膨胀。每个公开数据来源仍保留其原始许可证和署名要求，
项目许可证不会覆盖或重新许可第三方数据。

<!-- AUTO_RESULTS_START -->
### 正式运行自动摘要

尚未完成正式远端运行。脚本完成后会自动用真实训练元数据更新本节，不会填造结果。
<!-- AUTO_RESULTS_END -->
