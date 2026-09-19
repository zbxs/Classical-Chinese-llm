# DPO repair v1：显式参考策略与真实错误偏好实验

## 1. 目的与结论

本实验回答两个具体问题：

1. 旧 DPO 退化是否由 PEFT 场景下 `ref_model=None` 导致参考策略不是冻结 SFT 引起；
2. 使用修复后 SFT 的真实生成错误，而不是规则合成负例，能否稳定提高古译今能力。

结论是否定的。显式参考策略工作正常，训练偏好指标也正常上升，但两个 checkpoint 在独立冻结翻译集上都低于 DPO 起点。checkpoint-128 还重新出现不停止和重复，因此所有 DPO 权重均被拒绝部署，聊天服务继续使用能力修复阶段选出的双后端方案。

## 2. 参考策略修正

`src/classical_llm/training/dpo.py` 新增可选配置 `reference_model_name_or_path`。配置后，训练程序会单独解析并加载参考模型，将其全部参数冻结并切到 eval 模式，再显式传给 `DPOTrainer`。

正式训练前运行 `scripts/dpo_reference_smoke.py`：16 对数据、2 个更新步骤。第 1 步 policy/ref 完全相同时 loss 为 0.6931、reward margin 为 0；第 2 步 reward margin 上升到约 0.0305，验证了冻结参考分支和策略更新链路。

## 3. 偏好数据如何生成

起点是 `outputs/repair-capability-v2/base_plain/best`，即能力修复实验中被接受的 `Base→clean plain SFT`。源样本来自 HistoryTrans 的既有哈希切分，DPO 不接触冻结 test split。

生成步骤：

1. 从 repair train/validation split 固定随机种子 `20260918` 抽取 3,072/256 条候选；
2. 用起点模型在 BF16 下贪心生成，最多 128 个新 token，`repetition_penalty=1.05`；
3. 以数据集参考译文作为 `chosen`，模型实际回答作为 `rejected`；
4. 排除未正常停止、含替换字符、空回答、与 chosen 完全一致、四字重复率高于 10%、chrF 不在 3–68 的样本；
5. 按 rejected chrF 分为 low `<20`、medium `20–40`、hard `>=40`，目标比例 30%/50%/20%；
6. 得到 2,048 对训练数据和 192 对验证数据。

| split | 候选 | 合格 | 最终 | low | medium | hard | rejected 平均 chrF |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3,072 | 2,923 | 2,048 | 614 | 1,024 | 410 | 28.18 |
| validation | 256 | 247 | 192 | 67 | 94 | 31 | 26.95 |

数据文件为 `data/repair-dpo-v1/train.jsonl`、`validation.jsonl` 与 `audit.json`。每行保留 prompt、chosen、rejected、源样本 ID、生成来源、chrF、源文本复制相似度、停止/乱码/重复统计，便于复核和重新筛选。

## 4. 训练配置

策略和参考模型都从同一个 repair SFT adapter 初始化；参考分支冻结，策略分支用 QLoRA 更新。

| 参数 | 值 |
|---|---:|
| GPU | RTX 4090D 24GB，单卡 |
| beta | 0.05 |
| loss | sigmoid |
| learning rate | 5e-6 |
| max steps | 128 |
| micro batch / accumulation | 1 / 16 |
| max length | 512 |
| eval / save interval | 64 |
| dtype | BF16；4-bit QLoRA |
| runtime | 1,319.8 秒（约 22 分钟） |
| train loss | 0.5827 |

checkpoint-64 的验证 loss 约 0.5565、偏好准确率约 96.88%、reward margin 约 0.311；checkpoint-128 的验证 loss 约 0.5198、偏好准确率约 96.35%、reward margin 约 0.413。trainer 按 eval loss 选择 checkpoint-128，但下游评测表明它不是能力最优 checkpoint。

## 5. 独立下游结果与拒绝决定

三者使用完全相同的 128 条冻结古译今测试和 20 条基础通用题，BF16 贪心推理、最多 128 个新 token。

| 候选 | 翻译 chrF | 正常停止 | 替换字符 | 平均四字重复 | 基础题通过率 |
|---|---:|---:|---:|---:|---:|
| 起点 repair SFT | **27.01** | **100%** | **0%** | **0.27%** | 80% |
| DPO checkpoint-64 | 24.36 | **100%** | **0%** | 0.55% | **85%** |
| DPO checkpoint-128 | 22.11 | 97.66% | **0%** | 2.25% | **85%** |

checkpoint-64 虽保持正常停止，但翻译 chrF 下降 2.65；checkpoint-128 下降 4.90，且 3/128 条回答未正常停止，平均重复率明显上升。人工抽查还能看到无依据补充人物身份、官职和事件的幻觉。因此实验状态为 `complete_rejected`，没有替换现有聊天服务。

## 6. 如何解释负面结果

验证偏好准确率高而下游能力下降并不矛盾。DPO 优化的是给定 chosen/rejected 对的相对概率；当前 chosen 是单一数据集参考译文，其中存在省译、增译和疑似错配。按 chrF 选择难例又可能把“措辞不同但合理”的模型回答当作 rejected。模型可以更好地记住这种排序边界，却牺牲开放输入上的忠实翻译与停止行为。

因此本实验排除了“只要修正 ref model 就会恢复”的解释，但没有证明 DPO 方法本身无效。下一轮若继续，应先人工审核一批高质量语义偏好对，增加事实一致性与源文覆盖判据，并将冻结下游评测设为训练外的硬门槛；在此之前不继续扩大 DPO 或 GRPO 训练规模。

## 7. 复现与审计

核心入口：

```bash
source scripts/activate_remote.sh
python scripts/dpo_reference_smoke.py
python scripts/repair_dpo_v1.py
```

完整汇总见 `reports/generated/repair_dpo_v1.json`。服务器还保留两个 checkpoint、trainer state 与逐条生成结果；由于两个 checkpoint 均未通过接受标准，它们不作为推荐权重发布。
