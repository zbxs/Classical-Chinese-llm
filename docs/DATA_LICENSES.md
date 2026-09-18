# 数据与模型许可（发布审核中）

以下是实验配置声明，不构成重新授权。2026-09-18 固定版本 raw 页面自动读取失败；改读上游数据卡后确认 HistoryTrans 标记 MIT、TroyeML 标记 Apache-2.0、PoetryMTEB 声明 CC-BY-NC-4.0（链接如下）。这尚不能替代固定版本许可原文、署名以及聚合来源逐项核对；完整数据包目前用于私有备份，不直接公开推送。

| 来源 | 记录的许可 | 版本/定位 |
|---|---|---|
| [gujilab/chinese-classical-corpus](https://huggingface.co/datasets/gujilab/chinese-classical-corpus) | CC0-1.0 | 1979f15b4b749b3601fd9ee3e4111a725c079439 |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ODC-By-1.0 | cmn_Hani，main |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ODC-By-1.0 | sample-10BT，main |
| [HistoryTrans/Dataset](https://huggingface.co/datasets/HistoryTrans/Dataset) | MIT | 89087a1ea7ba831a34b9d3101281a6f1ff9334ea |
| [PoetryMTEB](https://huggingface.co/datasets/PoetryMTEB/Appreciation-of-Chinese-Classical-Poetry) | CC-BY-NC-4.0 | a44edb49f175381e6343b18d561e3d3e734b51ec |
| [TroyeML](https://huggingface.co/datasets/TroyeML/Ancient_Chinese_Study_252k) | Apache-2.0 | fcb5b55cc9b1d934063d833f31923b7e8acfc62d |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 需随基础模型许可核验 | 060db6499f32faf8b98477b0a26969ef7d8b9987 |

发布前须补齐对应许可原文/版权声明/署名；网页数据的底层内容权利以及聚合数据的来源须独立审查。PoetryMTEB 包含非商业限制，需保留署名、许可链接和修改说明，不能改标成自由商用。

修改方式包括文本标准化、筛选去重、哈希切分、分块混配、模板指令与规则负例。保留 acquisition.json、配置、manifest。偏好格式未复制 license 字段，须按 source_id 回溯，缺字段不代表无限制；无法回溯的样本不直接发布。

模型、输出和参考答案也需分别审查。自有代码许可证待所有者选择。本文件不是 LICENSE，私有备份不是已获许可的公开发行包。

PoetryMTEB 数据卡还列出出版物来源及 DeepSeek-V3.1 分析，不能把所有鉴赏参考答案称为人工金标准。TroyeML 是聚合数据，顶层 Apache 标签不能替代全部上游来源核验。
