# Classical-Chinese-llm：古文小模型全流程实验

以 Qwen2.5-0.5B Base 为起点，研究古文增量预训练（CPT）、指令微调（SFT）、奖励模型（RM）与 DPO / GRPO。正式实验在单张 RTX 4090D 24GB 上运行；Windows RTX 4060 Laptop 用于早期数据准备与冒烟测试。

## 当前状态与结论

**2026-09-18 修复进展：** EOS 小规模受控实验确认旧 SFT 不会正常停止；随后完成 Base/CPT/Instruct 三路线能力修复对照。`Base→clean plain SFT` 在 128 条冻结古译今测试上达到 chrF 27.01、100% 正常停止、0 替换字符，优于官方 Instruct 基线的 23.09，也高于 `CPT→plain SFT` 的 25.96。领域 Instruct SFT 的两个 checkpoint 都出现少量不停止重复，未被接纳为默认聊天模型。当前演示采用双后端：官方 Instruct 负责通用聊天，本项目最佳 adapter 负责明确的古文翻译请求。详见 `docs/CAPABILITY_REPAIR_V2.md` 与 `reports/generated/repair_capability_v2.json`。

**2026-09-19 DPO 复现实验：** 修正参考策略加载后，以最佳 `Base→clean plain SFT` 为策略与冻结参考模型，从该模型真实错误中构造 2,048/192 对训练/验证偏好数据，再训练 128 steps。虽然验证偏好准确率达到 96% 以上，checkpoint-64 与 checkpoint-128 的冻结翻译 chrF 分别只有 24.36 和 22.11，均低于起点 27.01；后者还重新出现不停止与重复。因此该 DPO 被明确否决，未部署到聊天服务。详见 `docs/DPO_REPAIR_V1.md` 与 `reports/generated/repair_dpo_v1.json`。

**2026-09-19 偏好因果诊断与 Dr.GRPO 修复：** 固定 7B 裁判对 2,240 个偏好对进行正反顺序双评，共 4,480 次判断，位置一致率只有 59.96%，低于预注册的 70%；A 被选 2,583 次、B 仅 1,407 次。随后 200 对独立评分复核的一致率更低（38%）。因此这批自动偏好标签、DPO 因果消融和新 Reward Model 均被门槛阻止，没有通过降低门槛继续。改为从同一最佳 SFT 起点运行两条**不含偏好标签**的 64-step Dr.GRPO：参考代理控制组 chrF 27.011，未超过起点 27.013；约束感知组 chrF 27.233、停止率 100%、通用题 80%，通过预注册点门槛，但配对 bootstrap 95% 区间为 `[-0.216, +0.691]`，包含零，故仅作为人工盲评候选，不替换默认模型。详见 `docs/PREFERENCE_CAUSAL_PROTOCOL.md`、`reports/generated/preference_grpo_v1.json` 与 `reports/generated/preference_grpo_analysis.json`。

2026-09-17 23:12（北京时间）完成三组 CPT、SFT、RM、DPO、GRPO，以及 Base/CPT/SFT/DPO/GRPO 五模型自动评测。每个模型回答同一组 400 道题，共 2,000 条生成结果；另有 15 组困惑度结果。人工盲评尚未进行。

**这是包含负面结果的研究记录，不是已经验证有效的古文助手。** CPT 的通用中文困惑度改善，但古文困惑度没有改善；DPO 古文困惑度明显恶化；GRPO 与 SFT 接近。低训练损失、高偏好分类准确率不等于文学能力提高。不能将问题简单归因于数据量，需要继续核验数据质量、训练目标与量化/推理差异。

实际完成每组 50M token 的三组配比实验；200M/500M、40B 和 1.5B 参数模型实验尚未执行。早期说明保留在 `docs/HISTORICAL_README_PRE_FINAL.md`，其中状态和计划不代表最终结果。

## 1. 目录与产物

| 路径 | 内容 |
|---|---|
| `configs/data/production_*_sources.yaml` | 正式数据源、版本、采集上限 |
| `data/raw/production_cpt/`、`data/raw/production_sft/` | 标准化采集数据与 acquisition.json |
| `data/cleaned/production_cpt/` | 清洗结果、切分与 audit.json |
| `data/final/` | 实际训练/评测数据与 token 清单 |
| `configs/remote/`、`configs/generated/` | 服务器训练与选中 SFT 配置 |
| `src/classical_llm/` | 数据、训练、评测实现和统一 CLI |
| `outputs/<stage>/<run>/best/` | 导出适配器与 tokenizer |
| `outputs/<stage>/<run>/checkpoint-*` | 中间 checkpoint，服务器保留 |
| `outputs/logs/production_state.json` | 流水线状态 |
| `outputs/evaluation/` | 逐题输出、评分、盲评表、困惑度 |
| `reports/generated/results.json` | 实际训练元数据和最终自动指标 |
| `reports/generated/final_audit.json` | 输出 ID、数量和题型核验 |
| `data/preference-diagnosis-v1/` | 偏好裁判失败证据、位置偏差报告与无偏好 GRPO 数据清单 |
| `reports/generated/preference_grpo_v1.json` | 两条无偏好标签 Dr.GRPO 的训练、冻结评测和接纳判定 |
| `reports/generated/preference_grpo_analysis.json` | 128 条冻结样本的配对差值、bootstrap 区间和约束指标 |
| `transfer/` | 私有备份，不提交 Git，不等于公开发行包 |

路径说明不表示全部大文件已经上传 GitHub。数据、权重的公开分发须通过许可审查。

## 2. 数据如何获取与生成

### CPT 原始语料与筛选

正式来源为 gujilab/chinese-classical-corpus（古籍）、HuggingFaceFW/fineweb-2 的 cmn_Hani（中文网页）、HuggingFaceFW/fineweb 的 sample-10BT（英文网页）。来源、版本和许可声明保存在配置与采集清单。

执行文本标准化、长度/中文比例检查、重复行和连续重复字符检查、网页模板检查、替换字符检查、精确/近似去重；按稳定来源分组逻辑切分，种子 20260912。启发式过滤不等于专家审校，也不能保证跨来源没有同文或污染。

清洗保留 44,496 条：train 40,291、validation 2,038、test 2,167。来源贡献：古籍 4,239、中文网页 28,303、英文网页 11,954。拒绝原因见 `data/cleaned/production_cpt/audit.json`，多原因计数不能直接相加当作删除条数。

使用基础模型 tokenizer 分块并按类别 token 配额混配。以下是构建清单计数，不等同于训练中实际预测 token；三组共享源池，并非互不重叠的新数据。

| 古文占比 | 数据行数 | 古文 token | 中文 token | 英文 token | 总 token |
|---|---:|---:|---:|---:|---:|
| 5% | 64,435 | 2,500,000 | 38,593,750 | 8,906,250 | 50,000,000 |
| 20% | 59,796 | 10,000,000 | 32,500,000 | 7,500,000 | 50,000,000 |
| 50% | 50,439 | 25,000,000 | 20,312,500 | 4,687,500 | 50,000,000 |

### SFT：参考答案来源

正式 100K 不是让当前 0.5B 模型凭空生成的答案，而是复用公开对齐数据：

- HistoryTrans/Dataset：读取最多 40,000 对古今文，构造两个翻译方向，标准化采集行数 80,000。
- PoetryMTEB/Appreciation-of-Chinese-Classical-Poetry：按 ID 对齐诗文与分析，多角度构造鉴赏问题，标准化行数 27,870。多角度题不是独立作品；简短分析未必满足长篇鉴赏指令。
- TroyeML/Ancient_Chinese_Study_252k：读取上限 160,000 行，按任务标签和关键词筛选创作/鉴赏候选。采集量不等于最终训练量。

统一文本、检查非空和输入输出长度比例，按任务及输入输出哈希去重，套用固定模板，保留来源/作品/作者/许可和 split 字段。最终 SFT 为 train 100,000、validation 5,000、test 5,000。数据是 system/user/assistant messages，最后一段为参考答案。自动校验不代表人工逐条验收；教师合成代码及冒烟样本不意味着正式数据经过强教师生成和审核。

### 偏好数据：chosen / rejected

从 SFT 参考数据抽取 20,000 对，chosen 为已有参考答案；没有现成 rejected 时，规则修改否定、数词，遗漏句子，加入现代口语，生成泛泛赏析或截断答案，并记录 error_types。按 source_id 稳定哈希切分为 train 16,006、validation 2,003、test 1,991。

20K 是总量，不是训练集数量。RM/DPO 使用偏好对，GRPO 使用提示现场生成回答。负例可能容易区分，且不一定是真实的人类偏好。偏好验证内容源自 SFT 数据池，不是全新未见内容的能力测试。

### 评测数据

从标记 test 的候选抽取古译今、今译古、鉴赏、创作各 100。生成时去除末尾参考答案，评分时才读取它。split 字段不能代替跨来源语义泄漏核验。

困惑度实际计入古文 199 篇、中文 200 篇、英文 200 篇，分别预测 1,175,562 / 158,993 / 119,957 tokens。按不重叠 1024-token 分块计算 token 加权 NLL 再取指数；不是滑窗困惑度，不应与其他上下文协议直接比较。

## 3. 训练步骤与耗时

基础权重 `Qwen/Qwen2.5-0.5B`，快照 `060db6499f32faf8b98477b0a26969ef7d8b9987`。单 GPU、NF4 QLoRA、BF16、梯度检查点、梯度累积；LoRA rank 32、alpha 64。具体参数以 remote YAML 和 `results.json` 的 config 为准。

| 阶段 | 工作量 | 训练时间 | 训练损失 |
|---|---|---:|---:|
| CPT 5% | 50M 配额 | 1.77 小时 | 3.6009 |
| CPT 20% | 50M 配额 | 1.76 小时 | 3.7324 |
| CPT 50% | 50M 配额 | 1.75 小时 | 3.8981 |
| SFT | 100K，2 epochs，12,500 steps | 8.74 小时 | 2.6761 |
| RM | 16,006 对，2 epochs，2,002 steps | 0.72 小时 | 0.0198 |
| DPO | 16,006 对，1,001 steps | 3.80 小时 | 0.0678 |
| GRPO | 2,001 steps | 57.35 小时 | 0.0113 |
| Dr.GRPO 参考代理控制组 | 64 steps | 10.20 分钟 | 0.00027 |
| Dr.GRPO 约束感知组 | 64 steps | 10.51 分钟 | 0.00099 |

耗时为 Trainer 返回值，不包含全部准备、失败重试和评测；不同目标的 loss 不能横向排名。

CPT 按同一验证集 eval_loss 选择 5% 分支：约 3.5564，20% 为 3.5588，50% 为 3.6634；不是按最终测试集选优。SFT 接选中 CPT；DPO/GRPO 从同一 SFT 起点分支，RM 单独训练用于 GRPO，后者还组合规则奖励。

GRPO 初期 384-token rollout 因耗时中止，尚无可恢复 checkpoint，改为 128 后重新训练，旧日志保留。这是预算导致的协议变更，**没有已验证的“覆盖 99% 答案 token 长度”结论**。截断会影响实验。GRPO 未设置验证选优，`best/` 是最终导出而非经验证挑出的最佳 checkpoint。

## 4. 五模型自动结果

PPL 越低越好；评测以 BF16 基座加载 adapter，训练使用 4-bit 基座，并非数值完全一致。

| 模型 | 古文 PPL | 通用中文 PPL | 通用英文 PPL |
|---|---:|---:|---:|
| Base | 52.90 | 70.10 | 19.27 |
| CPT 5% | 54.02 | 40.34 | 20.91 |
| SFT | 71.26 | 103.57 | 25.83 |
| DPO | 423.20 | 223.48 | 29.60 |
| GRPO | 71.14 | 103.47 | 25.85 |

逐句 chrF 算术平均（0–100，文本相似度，不是正确率）：

| 模型 | 古译今 | 今译古 | 鉴赏 | 创作 |
|---|---:|---:|---:|---:|
| Base | 6.534 | 4.176 | 2.921 | 0.359 |
| CPT 5% | 2.444 | 2.535 | 1.434 | 0.471 |
| SFT | 6.003 | 5.465 | 3.387 | 1.290 |
| DPO | 5.261 | 5.173 | 2.385 | 0.646 |
| GRPO | 5.961 | 5.645 | 3.752 | 1.312 |

非空率均 100%，只表示生成了文本。生成协议：贪心、最多 384 新 token、repetition_penalty=1.05、输入上限 2048。创作/鉴赏具有多种合理答案，单参考 chrF 不能代替人工文学评价。未计算置信区间、未多种子重复，微小差异不代表显著提升。

### 4.1 能力修复对照（独立协议）

该实验不复用旧 100K SFT 结论。它从 HistoryTrans 的既有 split 中重新筛选 8,192/256/128 条古译今 train/validation/test，按内容哈希去重，过滤异常长度、低中文比例、特殊 token、网页痕迹和连续字符。三条训练路线统一为 256 steps、QLoRA r=32/alpha=64；Base 与 CPT 使用无聊天标记的纯文本提示并以 `<|endoftext|>` 结束，Instruct 使用官方聊天模板和较低学习率。统一 BF16 加载，评测 128 条翻译和 20 条基础通用题。

| 候选 | 翻译 chrF | 正常停止 | 替换字符 | 平均四字重复 | 基础题自动通过率 |
|---|---:|---:|---:|---:|---:|
| 官方 Qwen2.5-0.5B-Instruct | 23.09 | 100% | 0% | 0.04% | 75% |
| Base→clean plain SFT | **27.01** | **100%** | **0%** | 0.27% | 80% |
| CPT 5%→clean plain SFT | 25.96 | **100%** | **0%** | 0.17% | 80% |
| Instruct→领域 SFT（256步） | 25.77 | 98.44% | 0% | 1.60% | 85% |
| Instruct→领域 SFT（128步） | 25.63 | 98.44% | 0% | 1.60% | 85% |

人工审阅否决了两个 Instruct SFT checkpoint：均有 2/128 条翻译重复到 128-token 上限，128步版本还有一条通用题未正常停止。自动分数最高的 Base adapter 被定义为**翻译专家**而不是通用聊天模型。演示服务的 `smart_repaired` 只在请求明确包含“翻译/白话/现代汉语”等词时路由到该 adapter，其余请求交给固定版本的官方 Instruct；响应会显示实际后端。这个双后端部署是工程补救，不代表得到一个同时提高所有能力的单一模型。

### 4.2 修正参考策略后的 DPO 复现实验

旧 DPO 的 `ref_model=None` 在 PEFT 场景下可能退回“关闭 adapter 的原始 Base”，而不是冻结的修复 SFT。复现实验显式加载同一个修复 SFT adapter 作为冻结参考模型，并先用两步 smoke test 确认初始 reward margin 为 0、更新后为正。偏好负例不是规则乱改文本，而是修复 SFT 在 BF16 贪心推理中的真实错误；从 3,072/256 条候选中筛出 2,048/192 对训练/验证数据。

| 候选 | 翻译 chrF | 正常停止 | 替换字符 | 平均四字重复 | 基础题自动通过率 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| DPO 起点：Base→clean plain SFT | **27.01** | **100%** | **0%** | **0.27%** | 80% | 保留 |
| DPO checkpoint-64 | 24.36 | **100%** | **0%** | 0.55% | **85%** | 否决：翻译退化 |
| DPO checkpoint-128 | 22.11 | 97.66% | **0%** | 2.25% | **85%** | 否决：翻译、停止和重复均退化 |

训练本身看似正常：128-step 训练 loss 为 0.5827；checkpoint-64/128 的验证偏好准确率分别约 96.88%/96.35%，reward margin 分别约 0.311/0.413。它证明的是模型学会了这批 chosen/rejected 的排序，不证明语义忠实度提高。当前主要问题是参考译文本身可能含噪、用单参考 chrF 挖 hard negative 会偏向表面差异，以及仅用自动参考答案作为 chosen 会放大错配。两个 DPO checkpoint 均保留在服务器作审计，不作为推荐模型或默认服务。

### 4.3 偏好失败因果诊断与无偏好标签 Dr.GRPO

为区分“DPO 目标不适合”与“偏好标签不可用”，先对 repair DPO 的 2,048/192 个训练/验证对做独立质量门槛。固定 `Qwen2.5-7B-Instruct` revision，以贪心解码分别查看每对答案的正序和逆序版本。

| 裁判协议 | 对数 / 调用数 | 一致率 | 高置信对 | 预注册门槛 | 结论 |
|---|---:|---:|---:|---:|---|
| 成对 A/B 双顺序评审 | 2,240 / 4,480 | 59.96% | 482 | ≥70% | 拒绝：明显首项偏差 |
| 单候选独立评分双规约 pilot | 200 / 800 | 38.00% | 4 | ≥70% | 拒绝：绝对评分也不稳定 |

首轮只有 27 次 JSON 解析失败（约 0.6%），但 A/B 胜者计数为 2,583/1,407，同一对在翻转后仍输出 A/A 的有 554 对、B/B 只有 46 对；因此问题不是简单的解析错误。按分项分数重新计算胜者，一致率仍只有 59.87%。这表明当前 7B 自动裁判不能为细粒度古译今偏好提供可信金标准。按预注册规则，后续 DPO 消融和基于该标签的 Reward Model 均停止；项目只能得出“当前偏好构造和裁判协议不可用”，不能外推成“古汉语领域不适合 DPO”。

从该失败吸取的 GRPO 经验是：不再使用二元偏好或新 RM，而是在已冻结的 capability-repair v2 数据上比较可审计奖励。两组均从 `Base→clean plain SFT` 开始，使用相同 2,048/192 条 train/validation、相同 64 steps、`loss_type=dr_grpo`、`scale_rewards=false`、截断 completion 屏蔽和 4 个在线采样；只改变奖励构成。

| 候选 | 翻译 chrF | 相对起点 | 配对 bootstrap 95% CI | 正常停止 | 替换字符 | 四字重复 | 通用题 | 点门槛 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 起点：Base→clean plain SFT | 27.013 | — | — | 100% | 0% | 0.265% | 80% | 基线 |
| 参考代理控制组 | 27.011 | -0.002 | [-0.390, +0.385] | 100% | 0% | 0.223% | 80% | 未通过 |
| 约束感知组 | **27.233** | **+0.221** | **[-0.216, +0.691]** | 100% | 0% | 0.261% | 80% | 通过，待人工盲评 |

约束感知组降低单参考 chrF 权重，并加入数字/否定保留、源文照抄惩罚、非空、反重复和对称长度护栏。它在预注册的点估计标准下通过，128 条中相对起点 32 胜、77 平、19 负；但置信区间仍包含零，仅运行一个种子，而且精确照抄率从 2/128 变为 3/128，尚不能证明照抄惩罚达成目标。因此该 adapter 只进入人工盲评候选，默认聊天服务仍保持原双后端配置。

**直白解读：**

- 旧流水线 GRPO 只是与 SFT 接近，没有可靠的增益证据。
- 新的参考代理 Dr.GRPO 对照组与起点实质持平（-0.002 chrF），说明“把单参考重合当为主奖励”没有带来改进。
- 约束感知 Dr.GRPO 有一个小的正向信号（+0.221 chrF），并且没损害停止、替换字符和基础题指标；但提升小、置信区间包含零、照抄指标未改善，因此不能声称“GRPO 已证明有效”。
- 最终决策是：保留约束感知 adapter 供网页对比和后续人工盲评；不将它替换成默认生产后端。

两组无偏好标签 Dr.GRPO 的奖励配方如下：

| 变体 | 奖励及权重 | 用途 |
|---|---|---|
| 参考代理对照组 | 参考 chrF 0.65，非空 0.10，反重复 0.10，参考长度 0.15 | 检验以单参考文本重合为主的代理奖励 |
| 约束感知组 | 参考 chrF 0.35，数字/否定保留 0.20，源文照抄惩罚 0.15，非空 0.10，反重复 0.10，参考长度 0.10 | 降低单参考依赖，增加可审计的忠实性和格式护栏 |

### 4.4 网页演示的路由规则

网页不是由一个新模型同时处理聊天和翻译，而是一个可解释的双后端工程路由。只有选择 `smart_repaired` 时会自动路由；在下拉框手动选择某个模型时，请求会直接发给该模型。

| 最后一条用户消息 | 实际后端 | 提示模板 |
|---|---|---|
| 包含“翻译”、“译成”、“译为”、“今译”、“白话”或“现代汉语” | `translation_repaired`（Base→clean plain SFT） | 无 chat special tokens 的固定翻译提示 |
| 不包含上述词 | 固定 revision 的官方 `Qwen2.5-0.5B-Instruct` | 官方 chat template |

翻译路由会优先提取引号中最长的内容；若无引号，取不包含路由关键词的最后一行；再无法区分时，取首个冒号后的文本。然后构造 `任务：请将下列文言文准确翻译为现代汉语，只输出译文。\n原文：...\n译文：` 的纯文本提示。这是因为当前最佳翻译 adapter 从 Base 训练，不应强行使用 Instruct chat template。

手动选择“Dr.GRPO 约束感知版”或“Dr.GRPO 参考代理对照组”时，它们也使用同一纯文本翻译提示，不参与 `smart_repaired` 的默认自动路由。页面回答下方会显示 `后端 ...`，用于核对本次请求实际使用的模型。“三个可靠模型逐个对比”只对比自动路由、官方 Instruct 和 Base→clean SFT；Dr.GRPO 因尚未通过人工盲评，不列入“可靠”组。

## 5. 故障与局限

1. 首轮领域生成误读 messages，输入成为空值文本；旧生成评分作废，保存在 `outputs/evaluation_invalid_none_prompt_20260917/`。修复提示和参考提取后，五模型全部重跑，最终报告只汇总当前 evaluation 目录。
2. 盲评表也曾漏读 messages，题目为 null；2026-09-18 修复导出代码，保留旧表并重建题目，不改变现有回答与自动指标。
3. PyTorch 2.5 与 TRL 的 FSDPModule 导入名不兼容，采用单 GPU 导入兼容处理；不是 FSDP2/分布式支持。
4. 首轮 DPO 明显退化；修正参考策略并改用真实模型错误后仍退化，且自动裁判未通过位置一致性门槛。问题不能只归因于 adapter 参考加载，也不能据此宣称 DPO 在整个古汉语领域无效。
5. 约束感知 Dr.GRPO 只取得 +0.221 chrF 的单种子点提升，bootstrap 区间包含零，精确照抄率也未改善；它是待人工盲评候选，不是已证实的改进。
6. 规则偏好、启发式任务分类、单参考答案质量、奖励投机、量化和 128-token 截断都限制结论；人工盲评尚未完成，不虚构分数。

2026-09-18 保存权重核验：DPO/GRPO 的 336 个策略 adapter 张量全部与 SFT 不同，权重均有限；两者保存的 ref 分支 336 个张量与 SFT 完全一致。权重差 L2 范数分别约 3.074 和 0.310。因此不是简单的“没有保存任何更新”，GRPO 更新幅度较小；这仍不能代替运行时加载和训练目标的完整有效性核验。详见 `reports/generated/adapter_audit.json`。

## 6. 环境与复现

正式环境：Python 3.12、PyTorch 2.5.1+cu124、Transformers 5.17、TRL 0.29.1、PEFT 0.20，以环境记录为准。服务器 `.venv` 使用 `--system-site-packages` 只读复用镜像 PyTorch，新增依赖装在项目环境中，不是完全独立于系统包。缓存在 `.cache`，Windows 环境不能直接复制给 Linux。

首次安装需网络；先核对版本再执行：

```bash
bash scripts/07_remote_setup.sh
source scripts/activate_remote.sh
python -m classical_llm.cli doctor
python -m pytest -q
```

重新构造数据时先核对许可，再执行 acquire → prepare → mix / build-sft → build-preferences → build-eval / build-general-eval。参数见 `python -m classical_llm.cli <命令> --help`，正式来源配置、采集清单和 manifest 为复核依据。网页源采集使用过 main，再次下载可能不同；保存的数据更接近本次真实输入。

恢复 `data/final/`、下载固定快照权重并确认 remote YAML 路径后启动：

```bash
screen -dmS classical-production bash scripts/08_production_pipeline_remote.sh
```

该脚本从训练开始，**不自动下载或构建全部数据**。跳过 complete 训练阶段，未完成阶段按配置恢复最新 checkpoint。非空生成文件会被复用，所以变更模型/数据/代码后要先归档旧输出，不能直接复用。

模型、数据和依赖就绪后可离线训练。screen 与本机断开后继续；实例关机、到期或磁盘问题仍会中断。监督需要联网，训练不依赖监督连接。

汇总结果无需重训：

```bash
python -m classical_llm.cli report --output reports/generated/results.json
python scripts/update_readme_results.py
```

偏好因果诊断和无偏好标签 Dr.GRPO 使用独立版本目录，不覆盖旧 DPO/GRPO：

```bash
# 首轮裁判位置偏差复核（读取已归档输出）
python scripts/analyze_judge_position_bias.py

# 独立评分小样本；质量门槛失败时以非零状态退出
python scripts/judge_preferences_independent.py \
  --model .cache/judges/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28 \
  --pair-limit 200 \
  --task-output data/preference-diagnosis-v1/judge-v2-pilot/tasks.jsonl \
  --judge-output data/preference-diagnosis-v1/judge-v2-pilot/judge_output.jsonl

# 两条不含偏好标签的 Dr.GRPO；会复用完整 checkpoint
python scripts/grpo_reference_recovery_v1.py

# 对 128 条冻结输出做配对 bootstrap 与约束指标分析
python scripts/analyze_grpo_reference_results.py
```

首轮和 pilot 的失败输出是实验结果的一部分；不要删除、改名为“通过”或把 `min_consistency` 调低后继续训练。`preference_grpo_v1.json` 中的 `complete_accepted` 仅表示约束感知组通过预注册自动点门槛且可进入人工盲评，不表示已经部署。

人工评分填写 `blinded_review.jsonl` 各维度 1–5 分；评分人不应查看 key 文件。用 `summarize-review` 汇总，未评分行不会变成已评分结果。

## 7. 发布与备份

详见 `docs/DATA_LICENSES.md`。代码、第三方数据和模型是不同许可对象；含非商业限制的数据不能改标为无限制商用，采集清单的许可字段不能代替授权核验。

仓库自有代码采用 MIT License；第三方数据、基础模型、生成数据和 adapter 仍分别受各自许可约束，MIT 不对它们重新授权。源码、配置、审计清单和结果摘要可公开，未完成来源逐项许可核验的大数据与私有实验备份不自动上传。

`scripts/finalize_artifacts.py` 核验 400×5 输出、5 组评分、15 组 PPL，保留旧盲评表后补齐题目，生成 `transfer/experiment-backup-20260918.tar.gz` 和 SHA-256。含数据、best adapters、日志、评测及代码；不含中间 checkpoint、环境、缓存、SSH 辅助文件；不删除服务器原文件。基础权重需另取固定快照；adapter 配置含服务器绝对路径，迁移时须明确加载相同基础模型。
