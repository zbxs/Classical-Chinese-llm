# 能力修复 v2：Base / CPT / 官方 Instruct 受控对照

目标是区分 CPT 是否造成额外退化，并选择兼顾古文翻译与基本指令能力的候选。旧正式模型、数据及结果均保留。

固定数据只取 HistoryTrans 古译今；按原 split 选择 8,192 train、256 validation、128 test。执行长度、中文比例、特殊标记、异常重复、哈希去重等规则。规则筛选不能代替逐条人工校对。

四个候选：官方 Qwen2.5-0.5B-Instruct 固定版本基线；Qwen2.5-0.5B Base 使用纯文本指令格式微调；旧 CPT 5% 使用同一纯文本格式微调；官方 Instruct 使用聊天格式低学习率微调。Base 与 CPT 的训练条件一致，均不使用 im_start/im_end，回答以原生 endoftext 结束。Instruct 使用其原生聊天模板。

三个训练候选各 256 steps，batch 4、梯度累积 4、NF4 QLoRA；Base/CPT 学习率 5e-5，Instruct 1e-5。Instruct 分支不是 Base/CPT 严格同条件对照，而是可用性候选。

统一以 BF16 加载评测：128 条冻结翻译题、20 条简单通用指令。自动指标包括停止率、替换字符、四字重复、chrF 和可核验通用题通过率。通用题仅覆盖极基础能力，不是完整基准；chrF 也不代表语义完全正确。

自动门槛：停止率至少95%、无替换字符、平均重复不超过10%，且通用题通过率不低于官方 Instruct 基线10个百分点以上。通过门槛后仍必须人工审阅，才能接入聊天默认模型。

## 实际结果与人工决策

| 候选 | chrF | 停止率 | 替换字符率 | 四字重复率 | 基础题通过率 |
|---|---:|---:|---:|---:|---:|
| 官方 Instruct | 23.09 | 100% | 0% | 0.04% | 75% |
| Base→plain SFT | **27.01** | **100%** | **0%** | 0.27% | 80% |
| CPT→plain SFT | 25.96 | **100%** | **0%** | 0.17% | 80% |
| Instruct→SFT（256步） | 25.77 | 98.44% | 0% | 1.60% | 85% |
| Instruct→SFT（128步） | 25.63 | 98.44% | 0% | 1.60% | 85% |

`Base→plain SFT` 是古译今指标最佳且行为稳定的候选。CPT 在完全相同的后续数据和超参数下低 1.05 chrF，因此旧 CPT 对本任务没有带来收益。两个 Instruct SFT checkpoint 均有 2/128 条翻译重复至 128-token 上限；128步版本还在基础题中出现一次不停止。故人工审阅没有接纳任何领域 Instruct adapter 为默认聊天模型。

部署采用 `smart_repaired` 双后端：明确的古文翻译请求使用 `Base→plain SFT`，其他请求使用固定快照的官方 Qwen2.5-0.5B-Instruct。该路由避免用领域微调 adapter 冒充通用模型，也不把官方基线的能力算作本项目训练收益。完整逐条输出位于 `outputs/repair-capability-v2/*.jsonl`，机器可读报告为 `reports/generated/repair_capability_v2.json`。
