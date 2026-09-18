# Translation Repaired Adapter

这是能力修复 v2 人工审阅后保留的唯一训练权重：`Base→clean plain SFT`。它是古文译现代汉语的单轮翻译专家，不是通用聊天模型。

- 基座：`Qwen/Qwen2.5-0.5B`
- 固定快照：`060db6499f32faf8b98477b0a26969ef7d8b9987`
- 方法：NF4 QLoRA，rank 32，alpha 64，256 steps
- 数据：`data/repair-capability-v2/` 中 8,192/256/128 条 train/validation/test
- 评测：128 条冻结测试，chrF 27.01，停止率 100%，替换字符率 0%，平均四字重复率 0.27%

加载时使用 PEFT adapter，并采用与训练一致的纯文本提示：

```text
任务：请将下列文言文准确翻译为现代汉语，只输出译文。
原文：学而时习之，不亦说乎
译文：
```

回答以基座原生 `<|endoftext|>` 结束。不要对这个 adapter 使用 Qwen chat template；通用聊天请使用固定版本的官方 Qwen2.5-0.5B-Instruct，或运行 `scripts/model_chat_server.py` 的 `smart_repaired` 路由。

完整协议、负面结果和人工否决理由见 `docs/CAPABILITY_REPAIR_V2.md`。adapter 是基座的增量权重，不包含 Qwen 基础权重；使用时还需遵守基座模型许可。
