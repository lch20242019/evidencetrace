# SciFact Agent A/B 空输出事故记录

## 状态与范围

- 最新单次 Judge 验证（2026-09-05）：已将冻结 OOF 证据接入真实生产 Judge，通过本地 Qwen 自由生成完成 30 条 train 诊断及 3 个合成样例。主切片 22 次生成中 15 次未通过生产校验；存在明确的无关证据错判支持，也有引用/输出契约问题。未调用 DeepSeek，未接入完整多 Agent；详见 [本地单次 Judge 诊断](scifact-single-judge-local-diagnostic.md)。
- 状态：生成/解析与冻结门禁缺陷已修复；Qwen 的 gold 证据关系判断诊断通过。旧单阈值 selector 的 train OOF 因完整证据召回不足而 `reject`；新双阈值完成 809 条 train OOF、14 项门禁 `pass`，已导出并冻结上游。随后完成固定 OOF 证据下的本地 Qwen 角色消融，外部官方评分核验通过，但额外复核低于单次判断。未接入原完整 Agent 工作流、未复测 dev，也未证明多 Agent 增益，不能宣布端到端事故全部解决。
- 发现日期：2026-09-04。
- 本次更新：2026-09-05；补入双阈值 train OOF、上游冻结与本地 Qwen 角色对照负结果，相关测试合并 105 项通过。双阈值校准复用缓存、无新增 NLI 推理；后续角色对照另执行了真实 Qwen GPU 前向，两轮计算量不混记。
- 影响范围：本地 `Qwen2.5-1.5B-Instruct` 的 SciFact single-agent / role-separated challenger 对照。
- 不受影响：官方 TF-IDF 复现、官方 VeriSci 复现、官方评分器封装、独立评分交叉校验、逐例 journal 哈希链。
- 旧产物策略：v3 运行作为不可变负结果保留，不覆盖、不删除，也不作为简历中的效果提升。

## 事故结论

正式 v3 运行完成了 SciFact 官方 dev 的 300 个 claim 和 900 次物理模型调用，但三个逻辑分支最终都生成了 300/300 个空 `evidence`。因此三者全部官方质量指标为 0，`judge_challenger - single_self_review` 的质量差值也是 0。

这不是评分器算错，而是 Agent 决策输出发生了系统性坍缩。运行与审计链路是可信的，策略结果是不合格的。

## 可复核证据

| 检查项 | 实测结果 | 证据 |
| --- | ---: | --- |
| 正式 dev 覆盖 | 300/300 claim，900/900 物理调用 | [`run_manifest.json`](../eval_runs/scifact_agent_ab_dev_frozen_v3/run_manifest.json) |
| 三个分支非空预测 | 均为 0/300；三份预测 SHA256 相同（`130b68d7…ff5c3`） | [`comparison.json`](../eval_runs/scifact_agent_ab_dev_frozen_v3_summary/comparison.json) 与三份 predictions JSONL |
| 主比较官方质量 | Self-review 与 Challenger 的 12 项官方指标均为 0 | [`comparison.json`](../eval_runs/scifact_agent_ab_dev_frozen_v3_summary/comparison.json) |
| 官方与独立评分 | 完全一致 | [`manifest.json`](../eval_runs/scifact_agent_ab_dev_frozen_v3_summary/manifest.json) |
| 初判格式有效 | 64/300（21.33%）；其中 61 个 `candidates=[]`、3 个只有 NEI、0 个关系候选 | [`comparison.json`](../eval_runs/scifact_agent_ab_dev_frozen_v3_summary/comparison.json) 与 [`raw_telemetry.jsonl`](../eval_runs/scifact_agent_ab_dev_frozen_v3/raw_telemetry.jsonl) |
| 最终格式有效 | Self-review 298/300；Challenger 300/300；600 次 final 均只生成 1 token | 同上；“格式合法”并未阻止语义为空 |
| 同预算 Token | 两个主分支均为 1,008,819 total tokens | 同上 |
| 组合路径 p95 | Self-review 14,671.14 ms；Challenger 14,680.42 ms | 同上；跨进程恢复后不报告整次 wall time |
| 官方 TF-IDF Top-3 | 188 个有证据 claim 中 142 个至少召回一个 gold 文档（75.53%） | [`metrics.json`](../eval_runs/scifact_official_tfidf_dev/metrics.json) |
| 官方 VeriSci 对照 | Abstract Rationalized F1 0.5000 | [`official_metrics.json`](../eval_runs/scifact_official_verisci_dev_score/official_metrics.json) |

至少 142 个有证据 claim 的 Top-3 中存在 gold 文档，但 Agent 仍对全部 300 个 claim 输出空证据。因此漏召回是上游限制，却不能解释本次 300/300 全空。

## 根因

### P0：输出协议允许“最短合法空答案”

最终调用使用 assistant prefill `{"evidence":{`。在 greedy decoding 下，模型只需生成 `}}` 就能得到格式合法的 `{"evidence":{}}`。实际 challenger 的 300 次 final 都生成该单 token，self-review 为 298 次 `}}` 和 2 次 `}}}`。协议没有独立的 verdict 字段，校验器只能确认 JSON 和引用范围合法，无法区分“经过判断后的 NEI”与“直接闭合的退化输出”。这是有直接遥测支持的主要机制；由于尚未完成独立的 schema/prefill 消融，不能声称它是唯一因果。

直接禁止空 evidence 不是正确修复：SciFact 的 NEI 和检索未命中场景本来就需要空 evidence。修复版显式输出根字段 `label`（`SUPPORT`、`CONTRADICT` 或 `NEI`）和 `citations`，校验一致性后再投影为官方 `evidence` 格式。

### P0：冻结门禁只验证完整性，没有验证行为和质量

旧 `write_frozen_config` 只检查 train 运行的 split、配置、完成状态和文件哈希。它不检查：

- train 样本量是否足以用于校准；
- 初判和最终输出有效率；
- 非空预测率是否坍缩；
- 官方 train 质量是否高于空输出基线；
- 官方评分与独立复算是否一致。

所以 12 条 train 校准全部为空仍能生成冻结配置。这属于门禁缺失，不是正常的负实验。

这 12 条中有 3 条带 gold evidence，且 gold 文档都已进入 Top-3：claim 2（CONTRADICT）、9（SUPPORT）、12（SUPPORT）。三个分支仍然 12/12 全空，说明 smoke 阶段已经出现了足以阻断冻结的反例。

### P1：校准集过小且只做 schema smoke

12 条样本只能发现基础接线问题，不能承担质量校准。更严重的是，本次 smoke 已经暴露全空，但流程没有 fail closed。正式冻结必须把“格式 smoke”和“质量 gate”分开：前者可用少量样本，后者必须使用足够规模的官方 train 数据及官方评分。

### P1：模型能力与检索上限

- 本轮模型是 1.5B 参数的通用指令模型，未针对 SciFact 训练；长科学摘要、数值冲突和精确句级引用对其能力要求较高。
- TF-IDF Top-3 的 gold document recall 为 69.38%，未召回的 claim 不可能由下游 Agent 找回证据。

这两项会限制最终上限，但不能作为全空输出的借口。必须先修复协议和门禁，之后才能测量模型与检索各自贡献。

### P1：生成参数被模型配置静默继承

旧 runner 没有显式传入或记录 `repetition_penalty`，实际从本地 Qwen 模型的 `generation_config.json` 继承了 `1.1`。train-only 消融表明，把它显式固定为 `1.0` 会改变输出分布，但不能单独恢复正确判断：移除 JSON 示例后，模型只是从恒定 NEI 转成恒定 SUPPORT，且 citation 结构失效。因此这是必须消除的实验变量，而不是完整的质量修复。

修复版必须同时在运行配置和逐调用 telemetry 中记录实际值，并由汇总器复核；不能只依赖模型目录的整体文件哈希间接绑定。

### P2：Windows 原生进程异常导致长任务中断

正式运行期间曾出现“指令引用了不可读内存”的 Windows 原生异常并中断进程。现有 Python 日志没有异常堆栈，已完成记录中的模型调用也没有 `generation_error`，因此目前只能确认是进程级中断，不能据此认定为某一行 Python 业务代码、CUDA 驱动或硬件故障。

本次没有通过重跑覆盖已有结果：runner 每个 case 都对哈希链 journal 执行 `flush + fsync`，随后用同一运行契约 `--resume` 校验并续跑，最终覆盖 300/300 claim。恢复机制已验证有效；若再次出现原生异常，需要另行采集 Windows 事件日志和 CUDA/驱动诊断，不能把推测写成根因。

后续 sentence selector 的第二次 train 启动中，Windows 事件日志记录了 `python` 的 `0xc0000005` 访问冲突，故障模块为 `msvcp140.dll 14.0`。改用 `msvcp140 14.44` 后，相同 tokenizer、模型加载与最小前向推理曾成功完成。这只提供一次可规避故障的观察，不能据此认定旧 MSVCP 版本是唯一根因。

2026-09-05 恢复核验发现，正常运行的既有 Python 3.9/PyTorch CUDA 环境也实际加载 `C:\Windows\System32\msvcp140.dll 14.0`，同时从 `D:\python` 加载 `vcruntime140` / `vcruntime140_1 14.28`。因此需要纠正“换成 14.44 就已定位并彻底修复”的过度判断：事件日志确认崩溃位置，尚未确定访问冲突的具体机制，也未证明历次内存报错同因。该既有 GPU 环境已完成 selector 的 v3 全量 train 校准，未出现新的原生崩溃。随后新增的固定环境入口与版本预检脚本也通过独立 synthetic GPU 验收；未复制 Office 等其他应用附带的 DLL。

### P2：解释器与 CUDA 环境不匹配

sentence selector 第一次 train 启动误用了 Python 3.11 的 CPU 版 PyTorch 环境，却请求 `--device cuda`，在设备预检阶段失败。这是已确认的启动解释器与依赖组合选择错误。后续使用既有 Python 3.9.2、`torch 2.2.0+cu118`、`transformers 4.49.0`、`scikit-learn 1.4.2` 环境完成 v3 校准；环境记录见其 [`run_manifest.json`](../eval_runs/scifact_sentence_selector_train_oof_v3/run_manifest.json)。本次运行完成与质量门禁通过是两件事，后者仍未达标。

## 同范围健壮性缺陷

- 旧 `_json_root_closed` 只累计容器深度，没有核对 `[`/`]` 与 `{`/`}` 的类型匹配，畸形括号可能触发过早停止。
- 旧 final parser 在集合成员判断前没有验证 label 是字符串，模型若返回 list/dict 可能再次触发 `TypeError`。
- invalid final 与合法 NEI 最终都写为空 evidence；评分时保留失败样本在分母是正确的，但诊断与冻结门禁必须分别统计，不能混为“模型判定 NEI”。

## 修复方案

### 1. 修复生成协议

- 初判和最终判定均使用根字段 `label + citations`，以 `{"label":"` 开始生成，避免在旧 `evidence` 对象入口处直接闭合。
- `label=NEI` 时必须为 `citations=[]`；`SUPPORT/CONTRADICT` 时必须至少引用一个范围内文档及其中一个范围内句子。
- 每个 citation 是 `[文档排名, [句子编号]]`；解析器验证后，将根 `label` 与对应句子投影为官方 `evidence`。生成协议没有独立的 `verdict` 字段。
- 保持严格 JSON、无语义修补、无重试、越界引用拒绝和失败样本留在分母。
- 旧 v3 parser/schema 不原地兼容；修复版使用新的 schema/hash/冻结配置和输出目录。

### 2. 增加 train 冻结门禁

- 冻结配置必须绑定 train run、三条分支预测及官方 train score 的哈希。
- 门禁至少检查：最小校准样本数、最终格式有效率、非空预测率、官方质量非零、官方与独立评分一致。
- 所有阈值和实测值写入 machine-readable gate report，并绑定进冻结配置。
- 任一条件失败即拒绝冻结；不能通过修改 dev 结果绕过。

### 3. 评测纪律

- 只在官方 train 上调整协议或阈值。
- train gate 通过后才能创建新冻结配置。
- v3 dev 已经被观察；修复后的再次 dev 必须标记为公开 dev 的重复、非盲运行，不能声称首次盲测。
- 只有同模型、同 Top-K、同上下文、同逻辑调用数和同 Token 上限的主比较才能评价角色分工。

## 验收标准

- 单元测试覆盖 `label/citations` 一致性及官方 `evidence` 投影、NEI 合法空输出、关系标签禁止空输出、label 类型错误、越界文档/句子和冻结门禁失败路径。
- train 质量门禁对 v3 全空产物必须 fail closed。
- 修复版 train 运行若仍全空或官方质量为 0，不得生成 dev 冻结配置。
- 运行器、冻结配置、官方 scorer 和汇总器的哈希绑定继续成立。
- 不把格式有效率当成回答质量，不把负结果写成多 Agent 提升。

## 修复过程记录

### 已完成的代码修复

- 新增 [`run_scifact_hf_agent_ab_v2.py`](../scripts/compat/run_scifact_hf_agent_ab_v2.py)：使用显式 `label + citations` 决策，严格校验 label 类型、关系与引用一致性、rank/句子编号范围、重复项，并改用类型匹配的 JSON 根闭合判断；旧 v1 runner 未修改，继续保留失败运行的源码哈希证据。
- v2 显式固定并记录 `repetition_penalty=1.0`，每次物理调用 telemetry 都写入 override；汇总器会复算并拒绝缺失或不一致的记录。
- 新增 [`freeze_scifact_agent_ab_config.py`](../scripts/freeze_scifact_agent_ab_config.py)：只接受完整官方 train 809/809、三分支官方评分与独立评分一致、格式率/非空率/最低质量通过的产物；两个主分支还必须至少出现两种最终标签，因而旧 v1 全空产物和全 SUPPORT 等常量分类都无法冻结。
- [`summarize_scifact_agent_ab.py`](../scripts/summarize_scifact_agent_ab.py) 同时保留旧协议审计并增加 v2 的 parser、生成参数、gate report、文件哈希与逐例重放校验。
- 相关 runner、冻结、汇总、官方评分协议和 train-only 诊断共 99 个定向测试通过，Ruff 通过。单元测试证明代码契约成立，不代表模型质量门禁已经通过。

### v2 train probe 01

- 产物：[`scifact_agent_ab_v2_train_probe_01`](../eval_runs/scifact_agent_ab_v2_train_probe_01/)。
- 新的显式 `label + citations` 协议使 initial 达到 12/12 格式有效；两个 final 分支各有 11/12 格式有效，证明自由文本 analysis、管道占位符和旧 prefill 的格式问题已被实质削弱。
- 但三个分支仍为 0/12 非空，claim 2、9、12 仍全部选择 NEI。结论是“格式层修复有效，语义单类坍缩未修复”。该 probe 不允许进入冻结门禁，也不会运行 dev。

### train-only 提示消融

- 在带 gold 文档召回的 claim 2（CONTRADICT）、9（SUPPORT）、12（SUPPORT）上，当前 v2 和仅重复 claim 的变体都输出 NEI/NEI/NEI。
- 去掉三类 JSON 示例并固定 `repetition_penalty=1.0` 后，输出变成 SUPPORT/SUPPORT/SUPPORT，且三条 citation 都不符合 schema。
- 即使把输入缩减到 gold sentence，claim 2 中“1/5 million”与“493/million”的明确数值冲突仍被判为 SUPPORT。
- 结论：JSON 示例既充当格式锚，也诱发类别捷径；删除/换序示例或重复 claim 只会改变常量类别，不能作为正式修复。下一版必须先定位证据，再进行受限关系判定，并由 train 门禁拒绝 constant-label 与 constant-empty 运行。

### train-only 语义后端可行性诊断

- 可复核产物：[`scifact_agent_nli_train_diagnostic_100_v1`](../eval_runs/scifact_agent_nli_train_diagnostic_100_v1/)；[`run_manifest.json`](../eval_runs/scifact_agent_nli_train_diagnostic_100_v1/run_manifest.json) 将其明确标为 `exploratory_not_generalization`，并绑定输入、模型、实现与 10 个输出文件哈希。
- 固定官方 train 文件顺序的前 100 条中，59 条有 gold evidence；TF-IDF Top-3 对其中 45 条召回 gold 文档（76.27%）。
- 直接对 Top-3 的全部 2,865 个 sentence-claim pair 取 DeBERTa NLI 最大分，会产生明显的 CONTRADICT 偏置；本轮探索的较优阈值下，官方式四项 F1 仍只有 0.310/0.276/0.192/0.181。
- 先用不读取标签的词法规则缩到 Top-10 sentence，再做 NLI，在同一 100 条诊断集上的标签准确率为 56%，官方式四项 F1 为 0.327/0.308/0.206/0.194；它打破了全空坍缩，但阈值已在这 100 条上探索，不能作为泛化成绩。
- oracle 诊断中，只对 gold rationale 做 SUPPORT/CONTRADICT 二分类时准确率为 81.82%，而加入 NEI 后三分类只有 48.76%。这说明该模型可作为“证据已定位后的关系判别器”，不能同时承担开放候选定位与拒答。
- 因此备用架构应拆成 `sentence selector → SUPPORT/CONTRADICT 二分类 → selector 无合格证据才 NEI`。同一个确定性 NLI 重复调用不能包装成多个 Agent，也不能据此声称多 Agent 增益。

### Qwen 的 gold 证据关系判断诊断

- 产物：[`scifact_qwen_oracle_train_gate_v1`](../eval_runs/scifact_qwen_oracle_train_gate_v1/)，指标见 [`metrics.json`](../eval_runs/scifact_qwen_oracle_train_gate_v1/metrics.json)，实现为 [`run_scifact_qwen_oracle_gate.py`](../scripts/run_scifact_qwen_oracle_gate.py)。仅使用官方 train 中 505 条有关系标签的 claim，其中 SUPPORT 332 条、CONTRADICT 173 条。
- 输入是从 gold rationale 确定性选出的证据，模型只判断 SUPPORT/CONTRADICT。采用 A/B 选项的下一 token 分数，并交换两次选项映射后平均；共 1,010 次前向推理，避免把自由生成格式错误混入该项能力诊断。
- 准确率 `0.7683168316831683`，二分类 Macro-F1 `0.7549367277613945`；固定预测多数类 SUPPORT 的对照 Macro-F1 为 `0.3966547192353644`，绝对差值 `0.35828200852603015`。SUPPORT/CONTRADICT 召回率分别为 `0.7620481927710844`、`0.7803468208092486`，达到预先定义的两类召回率均不低于 `0.60`、Macro-F1 比多数类对照至少高 `0.05` 的门禁，结果为 `GO`。
- 该结果仅证明“提供正确证据后，当前模型具备继续验证关系判断方案的能力”。由于直接使用 train 的 gold 证据，它不是检索成绩、泛化成绩或 Agent 端到端成绩，也不能据此声称空输出故障已全部解决。

### Sentence selector 代码与历史启动失败

- [`calibrate_scifact_sentence_selector.py`](../scripts/calibrate_scifact_sentence_selector.py) 已完成两轮独立审计及对应修订。冻结 NLI 模型，仅用可审计特征训练轻量分类器；按共享检索文档和重复 claim 分组，使用 nested OOF 与 lexical-only、NLI-only 对照比较。输入、模型和检索文件绑定哈希，阈值选择仅发生在 train 内部，正式推理前写入预注册文件。
- [`scifact_sentence_selector_train_oof_v1`](../eval_runs/scifact_sentence_selector_train_oof_v1/) 因 Python 3.11/CPU PyTorch 与 CUDA 请求不匹配而失败；[`scifact_sentence_selector_train_oof_v2`](../eval_runs/scifact_sentence_selector_train_oof_v2/) 因上述 Windows 原生访问冲突中断。两目录都仅有 `preregistration.json`，没有 NLI/OOF 指标、门禁结果或 `selector_model.json`；这两次启动没有可报告的 selector 效果，目录继续原样保留。
- 恢复核验后使用新的 v3 目录执行原定 train 校准，未覆盖历史失败记录。固定环境恢复入口与版本预检脚本在该次校准之后完成实现与独立验收，不能把 v3 描述为由新入口启动。

### Sentence selector 完整 train OOF 结果：拒绝冻结

可复核产物：[`scifact_sentence_selector_train_oof_v3`](../eval_runs/scifact_sentence_selector_train_oof_v3/)。[`run_manifest.json`](../eval_runs/scifact_sentence_selector_train_oof_v3/run_manifest.json) 记录：在既有 Python 3.9/CUDA 环境完成全部 22,266 个 sentence-claim pair 的一次 NLI 推理，以及 hybrid、lexical-only、NLI-only 各 809 条 OOF 预测，共 2,427 条；输入、源码和预注册在运行期间未改变。该次完整运行未出现新的原生崩溃；manifest 绑定的 12 项输出文件哈希已复核，未发现不一致。

以下句选择 F1 按官方兼容口径独立复算，不含 SUPPORT/CONTRADICT 标签正确性要求。该次没有调用外部官方评分器（`official_evaluator_invoked=false`）。数值来自 [`oof_metrics.json`](../eval_runs/scifact_sentence_selector_train_oof_v3/oof_metrics.json)，差值区间来自 [`cluster_bootstrap.json`](../eval_runs/scifact_sentence_selector_train_oof_v3/cluster_bootstrap.json)：对全部 228 个共享文档/重复 claim 连通分组配对重采样 10,000 次，种子为 `20260904`。

| 策略或比较 | 句选择 F1 | Hybrid 的 F1 绝对差值 | 差值的 95% CI |
| --- | ---: | ---: | --- |
| Hybrid | 0.28591065 | — | — |
| Lexical-only 对照 | 0.15666179 | +0.12924887 | [+0.09927473, +0.16970489] |
| NLI-only 对照 | 0.20124224 | +0.08466842 | [+0.05195233, +0.13505323] |

Hybrid 在 4/4 个外层验证折的句选择 F1 均高于两个对照，超过“至少 3/4 折不低于当折较强对照”的门槛；两项总体 F1 差值也均超过 `0.020`，且区间下界大于 0。但 [`acceptance_gate.json`](../eval_runs/scifact_sentence_selector_train_oof_v3/acceptance_gate.json) 仍为 `reject`，失败原因是完整 rationale 的召回不足：

| 验收项 | Hybrid 实测 | 预注册门槛 | 结果 |
| --- | ---: | ---: | --- |
| 全部关系 claim 的完整 rationale 召回 | 208/505 = 0.41188119 | ≥ 0.50 | 不通过 |
| Top-3 可召回关系 claim 的完整 rationale 召回 | 208/396 = 0.52525253 | ≥ 0.65 | 不通过 |
| NEI 正确空选率 | 234/304 = 0.76973684 | ≥ 0.70 | 通过 |
| OOF claim 覆盖 | 三策略分别 809/809 | 100% | 通过 |
| 跨折 claim / Top-3 文档交叉 | 0 / 0 | 0 / 0 | 通过 |

“完整 rationale”要求选中至少一组完整的 gold 证据句，不能用只命中其中一句替代。该结果说明混合特征比两个对照更有效，但当前配置仍遗漏过多完整证据，尚不满足后续 Agent 共用证据上下文的准入要求。按预注册失败策略，本次没有全量重训并导出 `selector_model.json`；目录核验确认该文件不存在。

这些数字属于官方 train 内部的 nested OOF 校准结果，运行记录明确标为 `nested_oof_official_train_only_not_generalization`。它们不是外部官方评分器结果、独立泛化成绩或多 Agent 效果，也不能与早期 oracle 诊断混作同一任务的提升。

### 固定 GPU 环境入口及独立预检验收

- 已新增 [`run_scifact_sentence_selector.ps1`](../scripts/run_scifact_sentence_selector.ps1) 和 [`check_scifact_gpu_runtime.py`](../scripts/check_scifact_gpu_runtime.py)，固定使用 `env\scifact-verisci-py39-gpu\Scripts\python.exe`；核验解释器、依赖、CUDA、模型及 selector 源码哈希，并在打开评测目录前执行 synthetic GPU 前向。运行时审计记录实际加载的原生模块；没有更改系统 DLL 或复制其他应用的运行库。
- 独立预检使用 `batch=8` 的自造句子对，输出 shape 为 `[8, 3]`、所有值有限，进程退出码为 `0`，不使用评测数据。审计为 [`scifact_sentence_selector_train_runtime_probe_v1.runtime.json`](../eval_runs/scifact_sentence_selector_train_runtime_probe_v1.runtime.json)，原生日志为 [`runtime.native.log`](../eval_runs/scifact_sentence_selector_train_runtime_probe_v1.runtime.native.log)。该次启用 `-PreflightOnly`，没有创建评测目录或启动新的校准。
- 这项预检发生在正式 v3 校准之后，验证新恢复入口的环境检查能力；v3 完整运行使用的是既有 GPU 解释器直接启动，不能倒写执行来源。入口相关 15 项测试、Ruff 和 PowerShell/Python 3.9 语法检查通过。

仅检查环境时，可在 `evidencetrace` 项目目录执行下面的命令。输出名必须未被使用且含 `train`；已有审计不会被覆盖。保留 `-PreflightOnly`，该命令不会触发校准：

```powershell
.\scripts\run_scifact_sentence_selector.ps1 -OutputDirectory scifact_sentence_selector_train_runtime_check_20260905 -PreflightOnly
```

### 当前语义修复门禁

- 早期自由生成和有限选项原型未能通过小样本证据定位检查；后续 Qwen 的 505 条 gold 证据关系判断门禁已通过。两者任务不同，后者不推翻前者的证据定位失败。
- 旧单阈值 selector 的 train OOF 因两个完整 rationale 召回门槛未通过而 `reject`；新双阈值轮已通过 selector 门禁。随后完成后文所述固定 OOF 证据下的受限角色对照，尚未接入原完整自由生成 Agent 流程；两种协议的质量门禁和能力范围不能混同。
- selector 或 Agent train 门禁失败时停止对应流程，不得降低冻结门槛或查看 dev 来“修复”结果。截至本次补录，本轮 train-only 诊断和 selector 校准未读取 dev 数据、未运行新的 dev 评分，也未产生新的 single-agent/multi-agent dev 对照结果。

### 旧单阈值轮的瓶颈与改动依据

- 对本次 train OOF 逐例独立复核，297 条未召回完整 rationale 的 claim 可分为：109 条 Top-3 没有 gold 文档；17 条已有 gold 文档但需要多句证据；171 条已有 gold 文档及单句完整证据却仍遗漏，其中 93 条选错或选不完整、78 条选空。因此，多句证据不是主要漏召回来源。
- 四折最终均选用 `threshold=0.7, max_sentences=1`。只读检查 [`nested_oof_audit.json`](../eval_runs/scifact_sentence_selector_train_oof_v3/nested_oof_audit.json) 已记录的 inner tuning sweep：每折 44 个候选配置，在 NEI 正确空选率不低于 70% 时，同时满足全部关系完整召回 ≥ 50% 和可召回关系完整召回 ≥ 65% 的配置数均为 0；四折最佳可召回关系完整召回为 62.78%–64.91%，均出现在 `threshold=0.7, max_sentences=4`。这只是现有候选网格内的诊断，不能承诺增加句数或调整选参排序就能解决问题。
- [`_configuration_key`](../scripts/calibrate_scifact_sentence_selector.py) 在满足 NEI 条件后优先优化自定义 best-rationale Macro-F1，而后置门禁还要求完整 rationale 召回；两者的优化目标存在差异。这是校准目标设计需要改进，不是已发现指标计算错误。
- 上轮仅提出 train 内联合校准与候选排序的后续方向，未实施新策略或重跑。下述新一轮用户授权仅覆盖“双阈值”和“选参目标对齐”，不修改候选排序；原 `reject` 产物和既定门禁继续保留。

### 双阈值与选参目标对齐：本轮方案

本轮只修改上述两个校准设计。使用独立的 [`calibrate_scifact_sentence_selector_v2.py`](../scripts/calibrate_scifact_sentence_selector_v2.py)，保留旧脚本和全部旧结果，不修改 TF-IDF Top-3、模型、特征、候选排序或 Agent 流程。

- **准入与扩展分离**：每篇文档最高句子分数达到 `accept_threshold` 才准入；仅在已准入文档中，使用 `expansion_threshold` 补选句子，且要求后者不高于前者。保留原全局排序及每条 claim 最多 1–4 句限制。两个阈值相等时应完全复现旧选择行为；降低扩展阈值不能让原本无合格证据的 claim 开始作答，也不能由一篇高分文档解锁其他低分文档。
- **约束先于目标**：train 内层校准先同时要求全部关系 claim 完整 rationale 召回至少 0.50、Top-3 可召回关系 claim 完整 rationale 召回至少 0.65、NEI 正确空选率至少 0.70；仅在可行配置中最大化官方兼容、无关系标签要求的句选择 F1。旧自定义 best-rationale Macro-F1 只报告、不参与选参。
- **明确不可行**：没有可行配置时记录 `selected=null` 与 `status=infeasible`；另外用不加上述约束时官方兼容 F1 最好的真实配置生成诊断预测，逐折和逐例标记，不能用全空输出替代弱基线。Hybrid 任一内层校准不可行即拒绝整轮模型导出。
- **固定实验条件**：两个阈值均沿用旧 11 个候选值；合法阈值对 66 组，乘以 4 个句数上限，每个策略共 264 个配置。保留原 4×3 嵌套分组 OOF、共享文档/重复 claim 防泄漏约束、随机种子、原外层验收和 10,000 次配对分组 bootstrap。
- **只复用固定特征**：核验旧 v3 manifest、全部输出、原始输入、模型及旧脚本哈希后，复用 22,266 条冻结 NLI 分数，重新构造并比对候选及静态特征；TF-IDF、scaler 和逻辑回归仍只在各自训练折拟合。本轮新增 NLI 推理为 0，不能把缓存来源的一次推理记作本轮调用。
- **先登记后运行**：新轮预注册绑定新旧源码、缓存、输入、模型与完整选参协议，运行结束再次核验。此前已观察过 train OOF，本轮是重复的 train-only 校准，不是新 holdout 或泛化成绩；无论通过与否，都不在本任务中启动 dev 或 Agent 对比。

本节保留运行前方案；实际验收结果在后文单独记录，不预先承诺通过。

方法依据：[scikit-learn 的评分指标选择说明](https://scikit-learn.org/stable/modules/model_evaluation.html)强调从最终任务目标选择评分函数；[决策阈值调优说明](https://scikit-learn.org/stable/modules/classification_threshold.html)区分模型分数与行动决策，并要求避免用同一份数据同时拟合模型和调整阈值。这些原则支持本轮目标对齐与内外层分离，但不保证本项目的具体网格能达到验收门槛。

### 双阈值实现与测试

- 新实现源码 SHA256：`056833466028590172edf036962fa356291c098c2c29517c293d9d757197cae3`；旧源码保持 `9d11dcc3575af86c33055b941cfaefb29b2d5ca287e09a15662a725a6ffe848a`，没有原地替换旧实验实现。
- [`test_calibrate_scifact_sentence_selector_v2.py`](../tests/test_calibrate_scifact_sentence_selector_v2.py) 在固定 Python 3.9 环境通过 **30 项测试**，包含阈值相等的旧行为等价、逐文档隔离、非法分数拒绝、三约束优先、不可行时禁止导出、真实合成 sklearn 嵌套 OOF 与 10,000 次 bootstrap、完整产物哈希和缓存篡改检查。合成测试不是 SciFact 效果结果。
- 旧 selector 与 GPU 运行入口共 **36 项回归测试**在现有项目测试环境通过；正式运行后，新旧三份测试文件联合执行 **66/66 通过**。新脚本及新测试 Ruff 与 Python 3.9 语法检查通过；新 PowerShell 入口通过语法检查、非法输出名拒绝及已有目录不覆盖检查。未为测试修改或安装环境依赖。
- [`run_scifact_sentence_selector_v2.ps1`](../scripts/run_scifact_sentence_selector_v2.ps1) 固定使用 `env\scifact-verisci-py39-gpu\Scripts\python.exe`；新脚本在拟合前核验 Python 3.9、scikit-learn 1.4.2、NumPy 1.26.4、SciPy 1.13.0，不加载 Torch/Transformers 进行推理。首次沙箱启动因 `WinError 5` 不能创建新结果目录而退出，尚未拟合、未留下结果目录；经授权提升写权限后使用同一命令启动，不属于模型或内存错误。

### 双阈值 train OOF 实际验收结果

运行产物：[`scifact_sentence_selector_train_dual_threshold_v1`](../eval_runs/scifact_sentence_selector_train_dual_threshold_v1/)。固定入口启动的完整运行退出码为 **0**，[`acceptance_gate.json`](../eval_runs/scifact_sentence_selector_train_dual_threshold_v1/acceptance_gate.json) 的 **14/14** 项检查通过。运行期间未出现新的原生内存错误；这不等于修复或排除了所有系统环境风险。

与旧单阈值 v3 使用相同 train、Top-3、模型静态特征及分组协议，对比数据分别来自两轮 `oof_metrics.json`。本轮同时改变了双阈值策略与选参目标，下表是两项改动的整体结果，不能单独归因于其中一项：

| Hybrid 指标 | 旧单阈值 v3 | 新双阈值轮 | 绝对变化 |
| --- | ---: | ---: | ---: |
| 官方兼容、无关系标签要求的句选择 F1 | 0.28591065 | 0.35345667 | +0.06754602 |
| 全部关系 claim 完整 rationale 召回 | 208/505 = 41.19% | 265/505 = 52.48% | +11.29 个百分点 |
| Top-3 可召回关系 claim 完整 rationale 召回 | 208/396 = 52.53% | 265/396 = 66.92% | +14.39 个百分点 |
| NEI 正确空选率 | 234/304 = 76.97% | 234/304 = 76.97% | 0 个百分点 |

新 Hybrid 共选出 1,029 句，官方兼容计数为 `relevant=1025, retrieved=1029, correct_selection=363`，对应句选择 F1 `0.35345666991236613`。这里的“完整 rationale 召回”是 claim 层面至少覆盖一组完整 gold 证据；不要与 pooled sentence recall `363/1025`、文档召回率或答案质量混淆。

逐例比较两轮 809 条 Hybrid OOF 记录：空/非空决策变化 **0 条**，旧版已选句被新版丢弃 **0 句**，新增完整证据命中 **57 条**，旧完整证据命中丢失 **0 条**。这与“保留准入阈值，独立补选证据”的预期一致；并非仅靠正确拒答总数相等就假定逐例决策相同。

**同轮基线仍保留真实预测**：Lexical-only 和 NLI-only 四折内层均无满足全部三约束的配置，因此每条预测都标明 `diagnostic_unconstrained_best`；没有把基线替换成全空。两者句选择 F1 分别为 `0.20900000000000002`、`0.22232472324723246`。Hybrid 相对这两个诊断对照的差值分别为 `+0.14445667`、`+0.13113195`；[`cluster_bootstrap.json`](../eval_runs/scifact_sentence_selector_train_dual_threshold_v1/cluster_bootstrap.json) 中 228 个分组、10,000 次配对 bootstrap 的 95% CI 分别为 `[+0.11473792, +0.17428352]`、`[+0.10648990, +0.16821902]`。Hybrid 在 4/4 个外折均高于两者；这些区间比较的是本轮 Hybrid 与本轮诊断基线，不是新旧 Hybrid 的差值区间。

**选参确实满足新约束**：[`nested_oof_audit.json`](../eval_runs/scifact_sentence_selector_train_dual_threshold_v1/nested_oof_audit.json) 保留每折每策略全部 264 组配置。Hybrid 各折所选配置及可行数如下（展示折号从 1 开始，JSON 从 0 开始）：

| 外折 | 准入阈值 | 扩展阈值 | claim 总句数上限 | 双阈值网格可行数 | 等阈值子网格可行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7 | 0.6 | 4 | 16/264 | 0/44 |
| 2 | 0.7 | 0.4 | 3 | 24/264 | 0/44 |
| 3 | 0.7 | 0.5 | 4 | 16/264 | 0/44 |
| 4 | 0.7 | 0.6 | 4 | 15/264 | 0/44 |

等阈值子网格复现单阈值行为；在相同内层 OOF 分数上，44 组等阈值配置每折均无可行项，而双阈值网格有可行项。这支持“本次阈值解耦扩大了可行策略范围”，但不是新数据集上的泛化证明，也不是额外调参后选出的结果。

门禁通过后，按预注册规则在全部 train 上重拟合并导出 [`selector_model.json`](../eval_runs/scifact_sentence_selector_train_dual_threshold_v1/selector_model.json)：`accept_threshold=0.7`、`expansion_threshold=0.55`（四折扩展阈值的中位数）、`max_sentences=4`（众数）。没有依据外层结果重新搜索最终阈值。上述质量指标属于四折 OOF 预测，不是该全量重拟合模型的独立测试成绩。

独立只读复核确认 **41 项 SHA256 绑定**一致，包括全部 **13 项新输出**；从 train gold 与实际选句重新计算三策略完整证据命中、NEI、句级计数和 F1，与存储指标及另一套官方兼容实现一致。全部 12 个“外折×策略”网格各有 264 组，所选 Hybrid 配置确为满足三约束后的既定排序最优项；32 项内外折 TF-IDF 拟合 claim 集与应有训练集完全一致。独立复算 228 分组、10,000 次 bootstrap 后，四项区间也全部一致。最终阈值和模型导出状态与预注册一致。

**任务边界**：此次只完成用户要求的双阈值与选参目标对齐及其 train 验收。三策略各有 809 条 OOF 预测，Hybrid 不使用诊断回退；新增 NLI 调用为 0，复用 22,266 条冻结特征。未修改 TF-IDF Top-K、排序模型或特征，未读取新 dev、未开展新 holdout、未集成 Agent，也未改写简历效果。完整证据召回虽然达到既定下限，仍只有 52.48%；不能描述为高召回、成熟可用或多 Agent 已优于单 Agent。

### 冻结上游与同模型角色对照：运行前协议

新一轮选择保留已通过 14 项门禁的双阈值 selector，不在同一轮更换排序模型。现有错误并未消失；但应先验证固定上游下的关系判断和角色复核，不能一边改上游一边归因 Agent 增益。若后续确认排序限制仍阻碍所需目标，再单独登记 SciFact 句选择器训练实验。参考 [VeriSci 官方模型说明](https://github.com/allenai/scifact/blob/master/doc/model.md)的检索、证据句选择、标签预测分层，训练折外验证必须重新拟合每个训练折，官方已在 train 上训练的权重不能直接用于无泄漏 train 评估。

本轮通过 [`freeze_scifact_selector_upstream.py`](../scripts/freeze_scifact_selector_upstream.py)绑定已通过 selector 的配置、权重、全部结果及原始输入；训练上下文只取 **809 条 Hybrid OOF 选句**，不把全量重拟合 selector 回代到 train。上下文只含 claim 和选中证据，不把 gold、NLI 极性概率或 selector 分数传给 Qwen。模型按原文顺序阅读句子；官方输出使用单独冻结的原 selector 引用顺序，避免前三句评分截断规则引入额外变化。

[`run_scifact_frozen_evidence_ab.py`](../scripts/run_scifact_frozen_evidence_ab.py)的新协议为：

- selector 选空时三臂均直接输出 NEI（空 evidence），不调用关系模型；非空时按文档使用本地 `Qwen2.5-1.5B-Instruct` 判断 SUPPORT/CONTRADICT。误准入和选错证据保留在评分分母；本轮下游不另做拒答。
- 复用已验证的 A/B 换序二分类评分，两个映射的语义概率取均值，固定平局规则；没有自由文本生成。一次共同初判后，分别由“同角色自查”和“独立角色复核”读取同一 claim、证据与初判标签；只有系统角色指令不同，独立复核允许保留原结论，不强制反对。
- 每篇被选中文档实际执行初判 2 次、两个复核各 2 次，共 **6 次物理前向**；两个主要比较臂各归属 **4 次前向**。共同初判在物理总量中只计一次；one-pass 是低预算诊断，不是公平主基线。
- 两臂均限制每次前向最多 **2,048 输入 Token**，不静默截断、不重试；实际输入 Token 可因角色指令长度不同而不同。读取下一 Token logits 不等于生成：实际生成 Token 为 **0**，被评分位置数另列。
- 使用固定 Python 3.9/CUDA 环境，模型加载后先做两次不含评测数据的 synthetic warm-up。逐 claim 交替两个复核臂顺序；每条完成记录写入哈希链并 `fsync`。当前不支持续跑，中断目录保留，不能将部分结果称为完整对照。
- 报告三臂官方兼容质量、Token、主臂共享初判加本臂复核的逐 claim 组合 p95、非空问题 p95、真实 GPU 推理时间。p95 不包含上游检索、selector、模型加载或磁盘持久化。API 实付为 0，本地计算费用未获计价依据时标为未计价，总成本不能写成 0。
- 运行前登记来源和协议哈希，结束时再次核对；正式产物需调用已复现的外部官方评分器，交叉核验全部三臂的 12 项指标。覆盖、预算、两类预测及最低非零质量构成 train 可用性检查，**不要求 Challenger 赢才能保留或报告结果**。

该实验只能检验“同模型角色提示对复核的影响”，不是完整自主多 Agent 协作效果；仍是重复 train OOF 上游条件下的诊断，不是新 holdout。本轮先完成冻结与 train 角色对照，不启动 dev。以上是运行前协议，实际结果另行补录。

### 冻结与角色对照实测：额外复核未带来收益

冻结产物为 [`scifact_selector_train_oof_frozen_v1`](../eval_runs/scifact_selector_train_oof_frozen_v1/)，绑定 42 项来源。独立重建确认 809 条 claim 中 379 条无选句、430 条非空，共 481 篇被选中文档、1,029 句；OOF 来源、原始引用顺序与所有上下文逐例一致。冻结文件 SHA256 为 `4d3a57c1c8955f2b4e26312c78f75d19f7f35ab56a69b01d85e9115358b69728`。

角色对照完整运行位于 [`scifact_frozen_evidence_role_train_v1`](../eval_runs/scifact_frozen_evidence_role_train_v1/)，退出码 0，809/809 条完成。三个外部官方评分目录分别为 [one-pass](../eval_runs/scifact_frozen_evidence_role_train_v1_score_single_one_pass/)、[self-review](../eval_runs/scifact_frozen_evidence_role_train_v1_score_single_self_review/)及 [challenger](../eval_runs/scifact_frozen_evidence_role_train_v1_score_judge_challenger/)；每个目录都包含官方和独立评分。三臂各 12 项指标一致，并与 runner 的独立实现相符。

| 策略 | 官方 Abstract Rationalized F1 | 官方 Sentence Label F1 | 输入 Token | 逐 claim 关系判断路径 p95 | 归属前向数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单次判断（低预算诊断） | 0.41913876 | 0.28919182 | 188,968 | 275.66 ms | 962 |
| 同角色自查（公平主基线） | 0.35789474 | 0.24829601 | 425,924 | 562.89 ms | 1,924 |
| 独立角色复核（公平主实验） | 0.35215311 | 0.24148004 | 426,886 | 565.01 ms | 1,924 |

主比较 Challenger − Self-review 的 Abstract Rationalized F1 为 **−0.005741626794258381（−0.57 个百分点）**，Sentence Label F1 为 **−0.006815968841285297（−0.68 个百分点）**。两者也都低于单次判断。没有把较差的结果替换、筛掉或改提示词后重跑；本轮没有事先登记显著性检验，不把小差值描述为统计显著。句选择完全固定，三臂 Sentence Selection F1 均为 `0.35345666991236613`，这不是角色复核的提升。

成本和性能口径：

- 完整评测实际执行 **2,886 次物理前向**，输入 **663,842 Token**，GPU 同步推理累计 **216.6972165 秒**；两个主要臂各归属 1,924 次前向，但共同初判必须从物理总量中去重。另有 **2 次 synthetic 预热**，单独记录、排除评测账目。
- 三臂 GPU 推理归属时间依次为 **70.1830 / 143.3995 / 143.4807 秒**；生成 Token 全部为 **0**，因为程序读取 A/B 下一词 logits，并没有生成这些数量的答案 Token。API 实付均为 **0 美元**；电费、硬件折旧或租赁价未取得，`local_compute_cost_usd` 与 `total_cost_usd` 均为 `null`，不能称总运行成本为零。
- 表内 p95 使用全部 809 条的“共同初判＋本臂复核”组合时间，空选句为零；只统计 430 条非空问题时，三臂 p95 依次为 **305.21 / 604.02 / 611.99 ms**。不含检索、selector、模型加载和持久化；这是本次单机顺序运行的观测，运行前期曾有约 10 秒的 CPU 回归测试并行，不是专用无干扰环境中的稳定性能基准，也不据此宣称微小延迟差具有意义。
- 模型启动出现 Transformers 对 eager 滑动窗口的提示；只读核对本地配置 `use_sliding_window=false`、窗口长度 32,768，本轮上限 2,048 Token。没有因此改动冻结配置，完整运行未出现新的原生内存错误。首次外部评分命令因传入归档哈希一字符误写而被校验拦截，未创建评分目录；更正命令参数后评分通过，没有重跑或改动模型预测。

新增两份测试 **39/39 通过**，连同此前相关测试合并 **105/105 通过**，Ruff 通过。新 runner 源码 SHA256 为 `64f19c4c669e23e4bd65df971cbd596c2b93fdce6461f255e17bafba4b72a41f`，冻结模块为 `7ff6615c3107ac4353e70bf68f5882f3ac63026ee8cc082947a617806d81d5d7`，旧脚本和结果未修改。运行 manifest 为 `03d4b398ef41998299390797eb898c6f656b8cd3cab7e8b312a8050feda0634b`。

独立只读审计再次核验了 809 条 journal 哈希链、42 项冻结来源、2,886 次调用的消息与实际 tokenizer 输入及 Token ID 哈希、双映射均值、初判传递、标签、引用和 0/6/4 调用归属；全部一致。p95、GPU 时间及账目重新计算一致，三臂共 36 项官方指标与外部评分、外部独立实现、runner 和逐例 gold/prediction 复算全部一致。审计未重新做模型前向、未读 dev，也未增加事后显著性筛选。

**判断与边界**：上游冻结与公平角色消融已完成，但本地 Qwen 的额外角色复核没有表现出收益。既然固定同一证据时单次判断反而更好，不能把问题全部归咎于上游排序，也不能仅为保留“多 Agent”叙事而宣称优化成功。本轮未训练新的 VeriSci 选择器、未复测 dev、未跑原项目的云端大模型或完整自主 Agent 工作流；这些本地结果不能代替它们的成绩，也不应写成简历中的多 Agent 提升。

**为什么本轮不需要 API Key**：所需权重已下载在 `env/models`，Qwen 在本机 GPU 上推理，selector 校准复用本地缓存并做本地拟合，没有请求模型服务商的云接口。单元测试使用合成数据验证代码契约，正式 809 条对照使用真实 SciFact 数据与本地模型，两者不同。无需 Key 不代表是假测试，也不代表无计算成本；若要评测项目实际使用的云模型，需要对应服务凭证并单独测量。

## 禁止误述

- 不能说“官方评分器坏了”或“TF-IDF 导致 0 分”。
- 不能把本轮写成“多 Agent 与单 Agent 效果相同”；两侧同时坍缩时，角色策略效果不可判定。
- 不能把 schema-valid rate 当成回答质量，也不能把修复后重复运行的公开 dev 称为未触碰 holdout。
