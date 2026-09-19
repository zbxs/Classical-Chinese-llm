# repair-dpo-v1 preference data

该目录记录修正参考策略后的 DPO 复现实验数据。

- `train.jsonl`：2,048 对训练偏好。
- `validation.jsonl`：192 对训练期验证偏好。
- `audit.json`：固定种子、生成模型、候选/筛选数量、分桶统计及审计样例。

SHA-256：

- `train.jsonl`: `eb28aa063fa5cad5579fe307a2e3d62a77ffdfa54d0a6fa2d07d131fb90a92d2`
- `validation.jsonl`: `fb195e0a6c7b45d0ceabbe41402a818a75adc0fa949ec9d75d639e6466e7a088`
- `audit.json`: `c50c2633587412125a9862ea3d9b4d8b71926798543c4ed3f1bf3f1369a74664`

`chosen` 来自 HistoryTrans 既有 train/validation split 的参考译文，`rejected` 是本项目 repair SFT 的 BF16 贪心生成结果。DPO 没有读取冻结 test split。详细生成与筛选协议见 `docs/DPO_REPAIR_V1.md`。

HistoryTrans 数据卡标记为 MIT；这份派生数据不改变上游许可和归属。使用前仍应核对 `docs/DATA_LICENSES.md`、上游数据卡以及具体样本的来源与质量。
