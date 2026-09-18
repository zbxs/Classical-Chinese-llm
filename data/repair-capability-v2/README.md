# Capability Repair v2 Data

本目录是能力修复 v2 的冻结数据切分：

- `train.jsonl`：8,192 条
- `validation.jsonl`：256 条
- `test.jsonl`：128 条

数据从 HistoryTrans/Dataset 的既有 split 中筛选，仅保留 `old_to_modern` 古译今任务；按文言文与译文内容 SHA-256 去重，过滤异常长度、长度比例、低中文比例、特殊 token、网页痕迹和连续重复字符。随机种子为 `20260918`，生成代码是 `scripts/repair_capability_v2.py`。

每行同时包含：

- `source_text` / `answer_text`：文言原文与现代汉语参考；
- `prompt_text`：Base/CPT 路线使用的无聊天标记纯文本提示；
- `messages`：Instruct 对照路线使用的 system/user/assistant 格式；
- `id`、`source_id`、`task`：追踪与去重字段。

上游采集清单声明 HistoryTrans/Dataset 为 MIT；见 `data/raw/production_sft/acquisition.json` 和 `docs/DATA_LICENSES.md`。筛选和自动检查不等于专家逐条校订，参考译文仍可能有省略、错译或风格差异。
