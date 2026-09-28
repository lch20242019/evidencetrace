# 冻结证据下的真实单次 Judge：本地诊断

日期：2026-09-05。

**同日后续状态：**原始 v1 诊断以下完整保留。随后仅补齐提示引用规则进行了同模型、同证据、同预算 v2 复测，生产校验失败 **15/22→12/22**，但官方 Abstract Rationalized F1 **0.193548→0.058824**，不构成效果改善。候选提示已隔离为默认关闭的实验选项；默认模型可见请求恢复原行为，运行时校验未放宽。详见[问题档案](resume-critical-issues.md)及[新原始报告](../eval_runs/scifact_local_production_judge_train_contract_v2/report.json)。下文“本轮未改提示”专指原 v1，不是最新代码状态。

结论：**本地 Qwen2.5-1.5B 在当前生产提示词和输出契约下不可靠，尚不能作为通过验收的单次 Judge。** 已发现无需归咎检索的语义错误，但错误同时包含输出约束和引用适配问题，不能声称全部由模型参数规模造成，也不能承诺换 DeepSeek 即可解决。

本轮只落实“先验证单次判断”。没有更改检索、排序、阈值或生产 Judge，没有加入 Challenger，没有读取 dev 或调用 DeepSeek，也没有改写简历指标。

## 实际调用路径与冻结范围

- 复用 [原冻结上下文](../eval_runs/scifact_selector_train_oof_frozen_v1/)，验证全部 **809 条** train OOF 来源；不把全量训练的 selector 回代到 train。
- 按 `SHA256(schema_version + ':' + claim_id)` 从小到大选 **30 条**，选择过程不看 gold、模型输出或难度。此切片已经属于反复使用的 train，不是 holdout，更不是正式泛化测试。
- 30 条中 **12 条选句为空**，经真实 Judge 的空证据分支直接拒答；其余 **18 条涉及 22 篇文档**，每篇最多一次生成。另有 **3 个合成诊断样例**，与主切片分账。
- 主程序使用项目既有 Python 3.11 环境；独立推理进程使用既有 Python 3.9/CUDA 环境。模型为本地 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- 实际路径为：[`ClaimJudgeAgent`](../src/evidencetrace/agents/judge.py) → 原 [`OpenAICompatibleClient`](../src/evidencetrace/model_client.py) 序列化 → 本地 JSONL 推理进程 → 原客户端解析 → 原 Judge 引用与规则校验。自定义传输层不发网络请求，不使用预制答案。
- 生成采用 greedy、BF16、eager、batch=1；每次最多 **4,096 输入 Token / 512 新 Token**，不截断输入、不重试、不补前缀、不修 JSON。本地没有服务端 JSON grammar，不能与提供约束解码的云 API 宣称完全相同解码条件。
- 输入只包含冻结选句。原文句号连续的选句用换行组成证据块，非连续句不拼接；所有分数为中性 `0`，不传 gold、selector/NLI 分数或额外文章正文。
- 生产 Judge 返回的真实引文按精确字符位置映射原句号；仅输出实际引用的句子，不能拿全部已选句冒充模型引用。部分句子的引用映射到句级评分，不表示模型引用了整句。

运行代码：[单次诊断入口](../scripts/run_scifact_local_production_judge.py)、[本地推理进程](../scripts/compat/scifact_local_judge_worker.py)、[冻结证据适配](../src/evidencetrace/eval/frozen_judge.py)。

## 实测结果

完整产物：[本地单次 Judge 运行](../eval_runs/scifact_local_production_judge_train_pilot_v1/)。

| 项目 | 实测值 | 口径 |
| --- | ---: | --- |
| 主切片完成 | 30/30 claim | 运行完成，不代表判断正确 |
| 主切片实际生成 | 22 次 | 每个已选文档一次，不是每个 claim 一次 |
| 通过生产校验的生成 | 7/22 | 仅表示格式、引用和规则校验通过，不是准确率 |
| 未通过生产校验 | 15/22 | 8 次 schema、6 次引用范围、1 次确定性冲突 |
| 完整 gold 证据已在输入中的文档 | 11 个 | 在冻结输入内至少有一组完整 rationale |
| 上述文档中通过校验且标签正确 | 3/11 | 包含格式/引用失败，不能写成纯模型语义准确率 |
| gold NEI claim | 7 条 | 5 条空选句、2 条非空选句 |
| 正确显式拒答 | 5/7 | 全部来自空选句规则分支，不是模型拒答能力 |
| 非空选句的 NEI 正确显式拒答 | 0/2 | 极小样本，只是问题线索 |

失败不会删除样本：官方投影对失败文档不输出证据，但保留全部 30 条 claim；独立拒答统计只把真正的 `not_in_source` 计为正确拒答，不能把 schema/引用错误变成“正确 NEI”。`partially_entailed` 等不直接冒充官方 SUPPORT。

主切片官方兼容评分：Abstract Rationalized F1 **0.1935483871**，Sentence Label F1 **0.1132075472**。此时标签和引用均来自真实 Judge，与旧二分类实验“引用全部由 selector 固定提供”的任务不同，不能用两轮分数直接计算优化增益。

原始 [report.json](../eval_runs/scifact_local_production_judge_train_pilot_v1/report.json) 同时保留其他评分、分母、逐例完整证据诊断和失败计数。[外部官方评分](../eval_runs/scifact_local_production_judge_train_pilot_v1_official_score_v2/) 已完成，四组共 12 个指标与内部实现一致；行序修复见下节。

## 错误归因：目前能与不能证明什么

**明确的语义拒答错误。** 合成样例 `900002` 的 claim 是 `The trial enrolled adults.`，唯一证据为 `The telescope observed distant galaxies.`。模型输出 `entailed` 并称后者直接支持前者。JSON、引用范围均通过，`finish_reason=stop`，不是检索没有找到正确文档导致的错误，也不是解析器把错误标签改成了支持。

另外两个合成样例的预期分别是支持、矛盾，均得到正确标签。因此结论不是“模型完全不能生成 JSON 或区分任何标签”，而是当前链路下的语义判断与拒答不稳定。三个样例是定位工具，不能包装成有代表性的准确率。

**不能统一归为语义能力差。** 8 次 schema 错误包括理由超过 `reason.maxLength=240` **3 次**、实质判断缺少引用 **3 次**、JSON 格式错误 **2 次**。6 次引用范围错误包括仅将组内换行替换为空格 **2 次**、跨不连续证据块拼接 **1 次**、复制 claim **2 次**、复制提示示例 **1 次**。这些失败不能都当作模型知识错误。所有原始请求、原始生成和结构化错误均保留，未偷偷清洗成成功答案。

**项目侧也存在明确契约缺口。** [`LiveJudgeSemanticOutput`](../src/evidencetrace/audit_models.py) 的运行时校验要求实质判断必须有非空引文；[`ClaimJudgeAgent`](../src/evidencetrace/agents/judge.py) 要求引用是单个证据块的精确连续原文。但前者的条件校验没有表达进发给模型的 JSON Schema，后者的精确字符串要求也没有被明确传达。字段的 `maxLength=240` 则已明确传达，超长不能归为完全没有告知。提示中另有固定正向 `entailed / 0.9 / Nimbus` 示例，实际发生示例文本复制；这提示可能存在示例干扰，但没有消融，不能断言示例就是语义错误的根因。

11 个完整 gold 证据文档中，**7 个是契约失败，4 个得到有效判定，其中 3 对、1 错**。claim `1084` 在有效输出中读反了风险方向；这提供了语义问题证据，但不能把总体 `3/11` 当作纯模型推理准确率。

本轮冻结了生产提示词和正向输出示例，没有做 prompt 消融，也没有强模型同输入对照。因此合理结论是“**本地模型＋当前生产提示词/契约组合需要改进**”，不是“已经证明根因全部是小模型”。先明确已有运行时契约，再在同样本、同证据上本地复测，是成本更低且更直接的下一步；本轮没有擅自修改生产提示词，也没有调用 DeepSeek。

## 性能、成本与运行情况

- 主切片输入 **13,209 Token**，真实生成输出 **2,049 Token**。全部 22 次都有完整用量记录；另 3 次合成调用不并入这两个主切片总数。
- 含合成样例的实际生成总数为 **25 次**，全部正常 EOS 结束，`finish_reason=stop`，最长输出 **192/512 Token**；不是输出额度截断造成的本轮失败。
- 30 条 claim 的观察性 Judge 路径 p95 为 **10,966.33 ms**，包含本地传输和生成日志持久化，不含上游检索、selector、模型加载；不是稳定的性能压测指标。
- API 费用为 **0**；本地电费、折旧和总成本没有定价依据，保持 `null`，不能称总成本为零。
- 推理进程和主运行均正常退出，推理进程退出码 **0**，本轮没有出现新的原生内存异常。这不等于修复了操作系统中所有潜在内存问题。
- 启动日志含 eager/sliding-window 与采样参数警告。既有模型配置未更改；当前 `do_sample=False`，模型默认的采样 temperature/top-p/top-k 不决定贪心采样结果。警告与本轮实际退出状态分别记录。

## 评测集成问题及修复边界

外部官方评分器要求 gold 和 prediction 的 claim 行序完全相同；新入口按 ID 哈希顺序执行并保存预测，gold 子集沿用原 train 文件顺序。内部实现按 ID 对齐可评分，但外部评分最初报 `Predicted claims do not match gold.`。

这是**本轮新评测导出入口的行序问题**，不是模型失败，更不是官方评分器故障。修复仅做预测行序规范化，要求 ID 无重复、集合完全一致、每个 ID 的标签和句号列表原样保留；不重新推理，不覆盖原运行或失败目录。评分前另有一次手写归档 SHA 参数错误，在创建输出前即被拒绝；随后直接读取历史已核验 manifest 中的 SHA，避免人工抄写。

已通过独立 [行序对齐工具](../scripts/align_scifact_prediction_order.py) 完成修复；[对齐记录](../eval_runs/scifact_local_production_judge_train_pilot_v1_order_alignment/manifest.json) 记录原输入与新输出哈希，30 个 ID 的预测内容逐一不变。新评分目录为 `scifact_local_production_judge_train_pilot_v1_official_score_v2`，失败目录原样保留。原入口产物仍需此对齐步骤才能交给官方评分器，没有事后修改冻结入口来掩盖错误。

独立审计核验了 13 项正式产物哈希、52 项源码绑定、42 项上游绑定及 25 条生成日志链。使用保存的原始 completion 做离线生产代码回放（没有再次推理），重现了 33 行主切片/合成结果、25 个生产请求、错误类别及官方投影。

相关回归测试 **209/209 通过**，包含本轮新增的 **66 项**适配、推理进程、诊断入口和行序对齐测试；这不是“全项目所有测试”的声明。新增实现与测试 Ruff 检查通过，推理进程 Python 3.9 语法检查通过。

## 后续边界

1. 不继续堆 Agent，也不根据这 30 条结果更改上游排序。
2. 先只明确提示词中遗漏的既有输出/引用要求，再本地复测；暂不同时改正向示例、模型、证据或预算，以便归因。不放宽引用校验，不清洗错误为成功；`not_in_source` 仍允许空引用，不能强迫无证据时给出引用。若仍有明显失败，再做 DeepSeek 同输入单次对照；报告提供方解码约束差异、Token、延迟和成本，不把供应商更换自动称为算法优化。
3. 本轮不参与简历效果提升表述。真实部署验收和未参与调参的独立测试仍需另外完成。
