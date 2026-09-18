# EOS 修复试验 v1

本试验独立于旧正式结果；旧数据与模型不覆盖。运行 `scripts/repair_pilot.py`，新输出在 `outputs/repair-eos-pilot-v1/`，数据在 `data/repair-eos-pilot-v1/`，状态与指标在 `reports/generated/repair_eos_pilot.json`。

## 修改与对照

- 修复 `文言文翻译` 被归为鉴赏，以及 `翻译成文言文` 的方向识别。
- 不再仅凭“古诗词/文言文/成语/解释”归入鉴赏；候选不足时应降低配额或补充合格来源，而非错分凑数。
- 有原始任务指令的 SFT 样本不再包裹更宽泛的要求，避免题材判断被扩写成完整鉴赏。旧 final 数据不原地重建。
- 为 SFT 增加显式 `assistant_end_token`，保存修改后的 tokenizer；现有配置不设置此字段时保留原模板。
- 机制实验两组从同一旧 SFT adapter 加载：control 用 im_end，eos_fix 用原生 endoftext。用户/system 的 im_end 不变。两组同种子、同数据、同学习率 5e-5、同 64 步、NF4、batch 4 × accumulation 4，仅改变 assistant 结束符及对应 EOS 设置。

## 数据与边界

仅从 HistoryTrans 的 old_to_modern 数据中按确定性顺序选择 1,024 train、64 validation、16 test；另有四道固定熟悉句子，仅作展示。按 source_id 和输入输出内容避免新划分之间重复；不声称彻底消除跨来源/预训练污染。

参考答案 12–180 字、总 token 不超过 384、排除特殊标记/URL/替换字符；训练上限 512，所以入选样本不应发生内容截断。只做规则筛选与少量人工抽查，不声称已人工逐条验收。抽查中仍有过度解释现象，因此此数据用于结束机制对照，不代表最终优质语料。

validation 留存，但本次不以其选 checkpoint；每组固定最后一步导出。16 条测试只进行小样本比较，不用于挑训练超参数后再声称独立测试。

## 验收与限制

原 SFT、control、eos_fix 使用相同 NF4 推理、贪心、repetition_penalty 1.05、上限 96 token。统计自然结束率（确实生成停止 token）、四字重复率、替换字符比例、16 条参考的平均 chrF，并查看四道展示题原文回答。

目标首先是结束率显著改善、复读减少；不能仅凭停止率认定翻译正确。替换字符比例不覆盖所有外文乱码，chrF 不是语义正确率。未通过输出审查时不扩大训练或启动 DPO/GRPO。展示到 BF16 聊天页面前需另行验证保存/重载与精度差异。

本脚本拒绝覆盖已存在的试验目录，异常时需保留现场并另行决定恢复方式。所有旧 checkpoint、数据及原聊天模型都保留。
