# 训练、选点和比较协议

> 历史目标协议，并非本次全部执行完成：本次仅 0.5B、三组 50M CPT；GRPO 修改了生成长度且无验证选优；未做人类盲评、多种子和置信区间。不可将以下公平性目标视为已经满足，实际参数和结果以 README/results.json 为准。

## 模型链

`Qwen2.5-0.5B Base -> CPT adapter -> SFT adapter -> {DPO adapter, GRPO adapter}`。奖励模型从相同 Base 建立序列分类头，并用同一 20K 偏好集合训练。每个阶段若输入目录不存在，只在配置明确提供 fallback 时回到 Base；正式实验前应移除 fallback 或检查日志，避免把未训练模型误标为阶段结果。

## 本地训练

所有阶段采用 NF4 双量化 QLoRA、BF16 计算、gradient checkpointing、batch size 1 和梯度累积。CPT 将文本预先 token 化并用 EOS 分隔后装块，避免 Windows 上没有 FlashAttention 时的跨文档扁平注意力。训练写入 `run_metadata.json`、trainer state、指标和 `best/` 适配器。

## Checkpoint 选择

只按固定 validation 集的 `eval_loss` 选择 checkpoint；训练 loss 仅作诊断。正式实验建议每 250 step 保存/评估，并保留最优与最近 3 个 checkpoint。模型规模/数据量关系用 0.5B 和 1.5B、50M/200M/500M token 的二维消融研究，除 loss 外还报告 token 吞吐、显存和领域/通用评测。

## DPO 与 GRPO 公平比较

DPO 从 SFT 模型出发，使用 chosen/rejected 和固定 beta。GRPO 从同一个 SFT 模型出发，每个提示生成相同数量候选；奖励由训练后的 RM、格式/约束分和防复制规则组合。两条路线必须使用相同 20K 问题、token 上限、随机种子与评测集。不能用 GRPO 训练时的 reward 当最终质量分。

## 评测与灾难性遗忘

领域评测每类 100 题：古译今、今译古、翻译鉴赏、诗词歌赋创作。自动指标包括 chrF、非空率和硬约束命中率；最终主结论使用匿名随机排序后的人工 1–5 分（建议双评审，报告一致性）。通用能力用训练前固定的中文、英文验证文本计算 token 加权困惑度，并补充公开通用基准。若领域分提升而通用困惑度显著升高或通用基准下降，即视作遗忘信号。

所有模型比较均同时报告均值、样本数、标准差/置信区间和失败案例；冒烟测试不用于算法优劣结论。
