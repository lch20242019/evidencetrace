# EvidenceTrace 简历项目说明：源码与原始产物核验版

## 核验规则

功能和技术只以 Python 源码、依赖配置为实现证据；README、架构说明和测试名称都不能单独证明功能。评测数字必须同时满足：

1. 有逐条原始 JSONL 或机器可读 JSON；
2. 指标算法能在源码中定位；
3. 可以从原始记录独立复算；
4. 明确开发集、评测单位和未覆盖范围。

**历史验证记录（不是 2026-09-05 最新运行状态）：**早期曾在现有 Python 3.11 环境执行与 CJK 检索、空召回退和评测链路相关的聚焦测试，结果为 **71 passed**；同时运行冻结数据构建、泄漏审计、v1/v3 离线检索对照和标准 runner。当时环境未设置 `OPENAI_API_KEY`，三个 live baseline 均被明确跳过，实际模型调用为 **0**。后续已进行本地模型评测，不能把这段历史“0 次调用”解释为项目从未调用真实模型；最新局部修复与限制见[问题档案](resume-critical-issues.md)。

随后在独立环境 `<local-env>/evidencetrace-scifact` 中完成 SciFact dev 适配器 v2、Top-5 语义 Judge 与严格评分链路，专项回归通过。正式运行覆盖 **300 个公开 dev claim、600 条逐路径预测**，使用固定 revision 的本地 MiniLM NLI 执行 **8,195 个 claim–证据子集 NLI pair、297 个 forward batch、0 次外部模型调用**；最大完整输入为 **486/512 tokens**，本次超限 NLI pair 为 **0**，推理禁止静默截断。评分器还会交叉校验数据包、run manifest、模型身份、逐行遥测、Top-5 检索结果及结构化证据。该运行仍是**公开 dev 非盲测、oracle cited-abstract、自定义 pair 诊断**，不是官方 test/leaderboard、不是 5,183 文档全库检索，也不覆盖 Claim Miner 或完整五角色链路。

## 简历可直接粘贴版本（投递强化版）

**EvidenceTrace｜可追溯技术文档事实核验 Agent**

**项目定位：**面向 Markdown/TXT 技术文档，构建从原子声明抽取、证据检索、关系判定到原文定位与人工复核的引用驱动核验工具，降低逐条核对事实与追查出处的成本。

**技术栈：**Python、Pydantic、Typer、HTTPX、PyTorch、Transformers、scikit-learn、Qwen2.5-1.5B、MiniLM NLI、pytest、SARIF 2.1.0。

**核心实现：**

- 基于 Python Controller 编排 Claim Miner、Coordinator、Scout、Judge、Challenger 五类角色，完成原子声明抽取、任务拆分、证据检索、关系判定和反方复核。
- 通过 Pydantic 严格 Schema、逐 Claim 调用预算、引用 offset 校验、单次重试和局部失败隔离约束执行链路，避免单个模型输出异常中断整批任务。
- 基于 MiniLM NLI 设计 Top-5 候选句组合判定与多句证据选择策略，将核验结论绑定到原文 locator 和字符区间。
- 部署 Qwen2.5-1.5B 本地关系判定路径，统一生成可供人工复核及工具集成的 JSON、Markdown 与 SARIF 2.1.0 结果。

**测试结果：**

- 固定三轮回归完成 **504 次逻辑模型调用**，首轮结构化输出成功率 **99.21%**，异常经单次重试恢复率 **100%**。
- 在 SciFact 公开 dev 的 **300 条 claim、8,195 个 NLI pair** 上，相对词法规则基线将三分类 **Macro-F1 从 0.271 提升至 0.586（+31.5pp）**、句级证据 **F1 从 0.090 提升至 0.360（4.0x）**，全程无需外部模型 API。
- Qwen2.5-1.5B 在 **505 条给定证据的支持/反驳样本**上取得 **76.8% Accuracy、0.755 Macro-F1**，两类召回率分别为 **76.2%/78.0%**，同步推理 **P95 约 198 ms**。

**30 秒口述：**这个项目把声明抽取、任务规划、证据检索、事实判断和反方复核拆成五个有明确契约与调用预算的角色，每项结论都绑定原文位置。为兼顾效果和本地部署，我分别验证了轻量 NLI 和 Qwen2.5-1.5B：MiniLM 策略在 SciFact 上把 Macro-F1 提升 31.5 个百分点，Qwen 在已给定证据的关系判定中达到 76.8% Accuracy；同时通过严格 Schema 和重试机制，使 504 次调用的最终结构化输出成功率达到 100%。

> **面试口径（不复制进简历）：**角色、日期和职责按真实经历另填。MiniLM 数字来自 SciFact 公开 dev 的非盲测、oracle cited-abstract 固定诊断，相对项目原词法/冲突规则基线；Qwen 数字来自官方 train 中 505 条已给定 gold evidence 的二分类关系门禁，并非自由生成、检索或端到端 Agent 指标。504 次结构契约统计来自另一组固定开发集回归，不能归因于 Qwen。产品定位是辅助人工复核，不是无人值守自动审计。

**备选量化口径（不与投递版混用）：**在 SciFact 官方 809 条 train 的折外校准中，双阈值相对单阈值将完整证据覆盖率从 **41.2% 提升至 52.5%（+11.3pp）**，正确空选率保持 **77.0%**。该数字来自重复使用 train 的 OOF，只能称训练阶段证据选择结果，不能称独立测试或泛化收益。

**不进入简历的负向验证：**2026-09-05 的引用提示候选方案在同一 30 条公开 train 样本上使生产校验失败从 15/22 降至 12/22，但官方 Abstract Rationalized F1 从 0.193548 降至 0.058824，因此已改为默认关闭的实验选项。详见[问题档案](resume-critical-issues.md)和[官方评分](../eval_runs/scifact_local_production_judge_train_contract_v2_official_score/official_metrics.json)。

源码：[产品流程](../src/evidencetrace/product.py)、[检索实现](../src/evidencetrace/retrieval/rank.py)、[语义 Judge](../src/evidencetrace/eval/scifact_judge.py)。公开 dev 产物：[严格指标](../eval_runs/scifact_public_dev_local_nli/scored/metrics.json)、[逐题评分](../eval_runs/scifact_public_dev_local_nli/scored/results.jsonl)、[独立复算](../scripts/verify_resume_metrics.py)。训练 OOF 留档：[单阈值](../eval_runs/scifact_sentence_selector_train_oof_v3/oof_metrics.json)、[双阈值](../eval_runs/scifact_sentence_selector_train_dual_threshold_v1/oof_metrics.json)。

### 主流开源参照（面试备查，不复制进简历）

最接近本项目判定任务的开源参照，是 SciFact 官方 VeriSci 的组件实验。[原论文 Table 3](https://aclanthology.org/2020.emnlp-main.609/)在公开 dev 上报告：给定 gold abstract 评估 rationale selection，给定 gold rationale 评估三分类 label prediction。它与本项目的“Top-5 自检索句 + 自定义严格证据集 F1”以及“仅支持/反驳的 gold-evidence 二分类”都不完全同口径。

| 系统与设置 | Rationale / Evidence F1 | Label Accuracy | Label Macro-F1 |
|---|---:|---:|---:|
| 本项目 MiniLM，Top-5 自定义诊断 | 0.360 | 0.593 | **0.586** |
| 本项目 Qwen2.5-1.5B，gold evidence 二分类 | 不评估 | **0.768** | **0.755** |
| VeriSci RoBERTa-large，SciFact 训练 | **0.721** | 0.757 | 未报告 |
| VeriSci RoBERTa-large，FEVER→SciFact | 0.697 | **0.819** | 未报告 |
| VeriSci SciBERT，SciFact 训练 | **0.744** | 0.692 | 未报告 |

这里最多只能说：Qwen 的 **76.8%** 与官方组件实验处于相近数值区间；由于二分类/三分类、train/dev、输入证据与训练方式均不同，不能声称超过 VeriSci。MiniLM 的 **0.586 Macro-F1** 也不能和官方 `Accuracy` 或 rationale F1 直接比较。

在更严格的完整流水线层面，[MultiVerS 论文 Table 2](https://aclanthology.org/2022.findings-naacl.6/)报告 SciFact 全监督 test 结果；其[开源仓库](https://github.com/dwadden/multivers)提供代码和 checkpoint：

| 开源系统 | Abstract-level F1 | Sentence-level F1 |
|---|---:|---:|
| VERT5ERINI | 0.682 | 0.634 |
| ParagraphJoint | 0.691 | 0.609 |
| MultiVerS | **0.725** | **0.672** |

这些指标同时要求标签与证据满足官方评分条件，而本项目尚无同口径的完整五角色 test 分数，所以不把它们做成“本项目 vs. 开源模型”的高低结论。`504` 次调用的 `99.21%/100%` 是本项目结构化输出可靠性统计，也没有与之协议一致的 SciFact 公共基线。

## 历史四维指标留档（不直接复制为当前简历成果）

以下保留旧实验数据和限制用于复核，不代表 2026-09-05 修复后的效果。自建集、oracle 上下文、规则基线与公开 train OOF 各有不同口径，不能混算提升或替换标签。

| 维度 | 基线 | 优化/实现 | 实测结果 | 可说到什么程度 |
|---|---|---|---|---|
| CJK 检索诊断 | 同一冻结 32 例数据上的 `lexical-cjk-ngram-v1` | `lexical-cjk-ngram-v3`：NFKC + CJK 二/三元倒排 + 中英候选合并 | test substantive（N=11）literal Recall@1 **0.9091→1.0000**、MRR **0.9394→1.0000**、Recall@5 **1.0000→1.0000**；全部 32 例空结果率 **3.125%→0%** | 只证明小样本字面证据排序诊断；两个正向差值的 95% CI 均含 0，不能声称稳定或泛化提升 |
| 空召回退 | 空/全零召回直接进入 Retrieval+Judge | 有预算的全文回退、零分标记、连续原文与字符/字节硬上限 | 4 个官方中文文档探针均先空召回、再 **4/4** 返回零分完整原文，**4/4** 满足 2,000 字符/8,000 bytes 上限 | 只证明策略不变量与接线路径，不是 live Agent 效果 |
| 效果 | Single Agent，72 例内部 holdout | Retrieval+Judge | fixed-six macro-F1 **0.7274→0.6138（-11.36 个百分点）**，预注册 gate 失败 | 证明门禁能发现退化；不能宣称检索或多 Agent 提升质量 |
| 效果回归 | Single Agent，另一组 72 例内部开发 pair，三轮 | Adaptive 路由 | macro-F1 均值 **0.9470→1.0000（+5.30 个百分点）** | 仅开发集回归；非盲测、未覆盖 Claim Miner |
| 效率 | 同一开发集 Single Agent，三轮 | Adaptive 路由 | 输入 Token 均值 **-36.22%**，模型调用均值 **-32.41%**，各轮 pipeline p50 的均值 **-12.59%** | 可证明按样本跳过无效模型调用；不是合并样本 p50、生产成本或线上 SLA |
| 工程 | 无自动恢复的首次模型输出 | 严格 schema + 最多一次重试 | **504** 次逻辑调用中首轮成功 **99.21%**；**4/4** schema 失败恢复，**0** 未恢复 | 可证明结构化输出恢复链路；不等于模型语义正确率 |
| 公开英文检索诊断 | 无独立检索优化对照；两条判定路径复用同一 `LexicalRetriever` 与 Top-5 | v2 官方句级索引 + 严格完整句计分 | SciFact dev 有证据子集（N=188）Full-sentence Hit@1/3/5 **52.66%/76.60%/87.77%**，Complete-rationale Coverage@5 **85.11%**，MRR **65.93%** | 只能给出当前句级检索的绝对能力；不能写成 NLI 带来检索提升 |
| 公开英文判定与证据 | 同一 Top-5 下的 Top-1 词法/冲突规则 | 本地 MiniLM 三分类 + Top-5 全部非空子集判定 + 精确多句证据 | SciFact dev（N=300）macro-F1 **0.271→0.586（+31.46 个百分点，95% CI [+24.41, +38.09]）**；有证据子集（N=188）项目自定义 Evidence F1 **0.090→0.360（+26.97 个百分点，95% CI [+20.62, +33.21]）** | 结果显示整套策略在这一固定公开 dev 上优于仓库规则基线；不能拆分归因，也不是官方 leaderboard 或五 Agent 效果 |
| 公开运行工程约束 | 测试 double/自报 manifest 可被误当正式结果 | reportable 门禁绑定 pack、模型、遥测和证据子集；完整输入超限即拒绝 | **600 条预测、8,195 个 NLI pair、297 个本地 batch、0 次外部调用**；最大 **486/512 tokens**，超限拒绝 **0** | 证明本次产物使用真实本地 checkpoint 且无静默截断；不是吞吐、成本或线上 SLA |

## 简历取舍依据

- [Harvard MCS](https://careerservices.fas.harvard.edu/resources/create-a-strong-resume/) 要求简历具体、基于事实、简洁且便于扫描；[UC Berkeley Career Engagement](https://www.career.berkeley.edu/prepare-for-success/resumes/) 建议在可行时量化结果，而不是为每个项目强行加入数字。
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) 要求最终评价使用独立测试集，并强调测试集应具有统计意义且能代表真实使用数据；[NIST AI 800-3](https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models) 进一步区分“在固定 benchmark 上的表现”和“对潜在新样本的泛化表现”。
- 两组 72 例数据不能混为一谈：v3_zh 是已消耗的内部 holdout，得到负向结果；phase3g 是作者自建、provisional、非 holdout 的 pair-level 开发集。二者都绕过 Claim Miner，均不足以证明完整五角色系统优势。正文并列呈现两者，防止只挑正向开发结果。

## 技术栈的源码出处

| 技术 | 源码或配置证据 | 直接证明的事实 |
|---|---|---|
| Python 3.11+、Typer、Pydantic v2、HTTPX | [pyproject.toml](../pyproject.toml) 第 5–16、25–26 行；[src/evidencetrace/cli.py](../src/evidencetrace/cli.py) 第 3、14–16、43、57–99、302–349 行 | 依赖版本和 console script 明确；CLI 源码实际注册 check/fix 命令。 |
| OpenAI-compatible JSON Mode、Pydantic 严格校验 | [src/evidencetrace/model_client.py](../src/evidencetrace/model_client.py) 第 794–805、824–829、948–965 行 | 将 Pydantic JSON Schema 放入提示载荷，请求的 response_format 实际为 json_object，并在本地调用 model_validate_json(strict=True)。因此不能写成供应商原生 JSON Schema Structured Output。 |
| SARIF 2.1.0 | [src/evidencetrace/sarif.py](../src/evidencetrace/sarif.py) 第 1、395–477 行 | 源码从 AuditArtifact 构造 SARIF，并固定 schema/version 为 2.1.0。 |

## 公开 SciFact dev 诊断（适配器 v2，2026-09-04）

### 为什么此前没有，为什么这次仍然要补

- 此前缺少公开集不是一个值得辩护的优点：只有自建中文数据，外部可比性确实不足。但原因也不只是“没配环境”。EvidenceTrace 的内部任务是中文技术文档、六类关系、包含 Claim Miner；SciFact 是英文科学 claim、官方原子 claim、`SUPPORT/CONTRADICT/NOINFO` 三类，两者不能直接等同。[SciFact 论文](https://aclanthology.org/2020.emnlp-main.609/)将其定义为科学声明的文献检索、关系判定和 rationale 识别任务；[官方数据格式](https://github.com/allenai/scifact/blob/master/doc/data.md)也明确给出 cited document 与多组 rationale 结构。
- [官方仓库](https://github.com/allenai/scifact)说明 test gold 不公开，本地可复算的是公开 dev，而不是官方 test。因此本次使用完整 **300 条 dev**，必须写成“公开 dev 非盲测”，不能写“公开测试集”或 leaderboard 成绩。
- 正确做法是双轨评测：SciFact 补英文公开数据的外部诊断，自建中文集合继续覆盖 CJK、六类标签和业务边界。用 SciFact 取代中文评测同样是错误的。

### 数据适配与可复现约束

- 官方 release 在本机实读为 **5,183 篇 corpus、809/300/300 条 train/dev/test claim**；本次完整保留 dev 的 300 个原始 claim，标签映射后为 **124 entailed、64 contradicted、112 not_in_source**。`not_in_source` 只表示 cited abstracts 中没有已标注证据，不等于全世界无法核验。
- 每题只生成一个 source，按 cited ID 顺序拼接已引用摘要；因此这是 **oracle cited-abstract** 诊断，不测 5,183 文档的全库召回。300 题中保留 **338 组 rationale alternatives、83 个多 rationale claim、10 个多证据文档 claim、9 组非连续 rationale**。
- 官方 dev claim 1245 重复列出文档 `7662395`。构建器在 sidecar 中保留原始 `[7662395, 7662395]`，渲染 source 时按首次出现顺序去重为 `[7662395]`，避免重复 sentence key；异常没有通过删题掩盖。
- v2 source 正文只保留官方 abstract 句子并以换行连接，不再注入 `[DOC]`、`[SENT]` 或 `Title:`；标题、doc ID、sentence ID、locator、字符区间与文本 SHA-256 放入无标签 [sentence_index.jsonl](../eval_sets/scifact_public_dev/sources/sentence_index.jsonl)。gold rationale 继续单独保存在 [gold_rationales.jsonl](../eval_sets/scifact_public_dev/gold_rationales.jsonl)，runner 的输入哈希明确不包含该文件。
- 数据构建见 [build_scifact_public_dev.py](../scripts/build_scifact_public_dev.py)，冻结包与哈希见 [manifest.json](../eval_sets/scifact_public_dev/manifest.json)。官方 claims/evidence、abstract 和代码的许可证分别按[官方 LICENSE](https://github.com/allenai/scifact/blob/master/LICENSE.md)记录为 CC BY 4.0、ODC-By 1.0 和 Apache-2.0。
- 运行器 [run_scifact_public_dev.py](../scripts/run_scifact_public_dev.py)对每题只执行一次 `LexicalRetriever(top_k=5)` 并让两条路径复用同一结构化结果。仓库控制组 `lexical_rules` 只看 Top-1：去停用词后的 claim 词覆盖率达到 0.75 判 SUPPORT；数字/日期/版本或否定不一致且覆盖率达到 0.25 判 CONTRADICT；其余判 NOINFO。候选 `retrieval_judge_scifact_nli` 使用本地 [cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) 对 Top-5 的全部非空子集作三分类，再输出逐句 locator 与字符区间。数字、实体等确定性信号在 NLI 路径中只作提示，不能单独强制矛盾。
- 正式 run 固定模型 revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` 与权重 SHA-256，并核对本地 Hugging Face 下载 metadata；这只是本地 provenance 一致性校验，不是对远端仓库的密码学认证。测试 double 会被标记为 non-reportable，默认评分器拒绝接收。
- 评分器 [score_scifact_public_dev.py](../scripts/score_scifact_public_dev.py)将 run 输入哈希绑定到当前 pack，并复核模型身份、逐行 NLI 遥测、同检索器、`predicted_evidence ⊆ retrieved_evidence`、句级 offset/locator 与 gold sidecar；之后执行 10,000 次 claim-level paired bootstrap（seed=`20260903`）。

### 本地严格评分结果（非官方 leaderboard 口径）

判定指标分母为全部 **N=300**；证据与检索指标只统计存在官方 rationale 的 **N=188**，112 个 NOINFO 不进入证据分母。

| 指标 | `lexical_rules` | `retrieval_judge_scifact_nli` | 解释 |
|---|---:|---:|---|
| Accuracy（N=300） | 0.4100 | **0.5933** | 绝对差值 **+18.33 个百分点**；多数类 Accuracy 为 0.4133 |
| Fixed 3-way macro-F1（N=300） | 0.2714 | **0.5860** | 绝对差值 **+0.3146**；95% CI **[+0.2441, +0.3809]** |
| Strict evidence sentence-set F1（N=188） | 0.0904 | **0.3601** | 绝对差值 **+0.2697**；95% CI **[+0.2062, +0.3321]** |
| Complete-rationale coverage（N=188） | 0.0904 | **0.4628** | 一组 gold rationale 的全部句子被覆盖；多余句由 sentence-set F1 另行惩罚，所以不称 exact match |
| Joint verdict + complete rationale（N=188） | 0.0585 | **0.3883** | 必须同时判对关系并完整覆盖至少一组 rationale |
| Full-sentence Hit@5（N=188） | 0.8777 | 0.8777 | 两条路径同检索器，数值相同是结构不变量，不是 NLI 带来的检索提升 |

同一检索器的绝对诊断值为：Full-sentence Hit@1/3/5 **0.5266/0.7660/0.8777**，Complete-rationale Coverage@1/3/5 **0.4894/0.7340/0.8511**，Full-sentence MRR **0.6593**。v2 的检索单元就是官方完整句，结构化 offset 与 locator 均校验通过，因此 overlap 与 full-sentence 数值相同；仍应写 Hit@K，而不是把句级命中率冒充文档库 Recall@K。

多数类基线的 Accuracy/macro-F1 为 **0.4133/0.1950**；NLI 候选分别为 **0.5933/0.5860**。在该固定公开 dev 上，结果显示当前“Top-5 子集语义判定 + 精确选证”整套策略优于仓库规则基线和多数类 sanity baseline，但不能证明每个组件各自贡献，也不能与官方全流水线或主流 SciFact leaderboard 横向比较。该 NLI checkpoint 只在通用 NLI 数据上训练，阈值也未用 SciFact train 校准；从最多 31 个子集中取最高分还存在 multiple-comparison 风险，当前绝对 Accuracy 只有 **59.33%**，仍有明显改进空间。

本次真实本地运行产生 **600 条预测**，共评估 **8,195 个 NLI pair** 并合并为 **297 个 CPU forward batch**，外部 provider calls 为 **0**。模型完整输入上限为 512 tokens，实测最大 486，`rejected_overlength_pair_count=0`；代码使用 `truncation=False`，任何未来超限输入会在 model forward 前 fail-closed，而不是把模型未看到的句子计为证据。

### 原始产物与哈希

- 冻结包 [manifest.json](../eval_sets/scifact_public_dev/manifest.json)：`59d30e14bc599aed22218e7b63ca2c26a1033e439cd75d8575967ad5331a19f7`；官方原始文件哈希与全部冻结输出哈希均记录在其中。
- [600 条逐路径预测（300 claims × 2 paths）](../eval_runs/scifact_public_dev_local_nli/eval_results.jsonl)：`be2e12c5020c5a764ee950bfde19fd4364df93bd11765ad68a52b7c2dd46bf84`；[run manifest](../eval_runs/scifact_public_dev_local_nli/run_manifest.json)：`7aef8870667dd9391d8c81ac31cb08caf9af1b96920c13fb35789d0cb794fde1`。
- [严格 metrics.json](../eval_runs/scifact_public_dev_local_nli/scored/metrics.json)：`4b4adccba6b94e9c75c9450921b5a83f16a0f5e2ae62c26176625e32ad31eb4f`；[逐题评分](../eval_runs/scifact_public_dev_local_nli/scored/results.jsonl)：`b92f886de4929b3ce7b90d1a3bd8bdf146f8db8e19a8ca3404e11bf4ffc87d9a`；[评分报告](../eval_runs/scifact_public_dev_local_nli/scored/report.md)：`41ac9cd3da44372ee8e5483197dcc789f045f363f5c54c0acf18aaf5dd6eadbc`。
- v1 markup 数据包与 pre-NLI 规则结果分别保留在 `eval_sets/scifact_public_dev_superseded_v1_markup/`、`eval_runs/scifact_public_dev_deterministic_superseded_v2_pre_nli/`，并有 `SUPERSEDED.md`；更早的 overlap 口径保留在 `eval_runs/scifact_public_dev_deterministic_superseded_overlap_v1/`。这些目录仅供审计，不得与当前结果混用。

**4 个 SciFact 专项测试文件共 45 项通过（非全仓测试）**，来自 [adapter tests](../tests/test_scifact_public_dev_adapter.py)、[semantic Judge tests](../tests/test_scifact_semantic_judge.py)、[runner tests](../tests/test_scifact_public_runner.py)和 [scorer tests](../tests/test_scifact_public_scoring.py)。它们覆盖无标签句级索引、gold 隔离、Top-5 的 31 个非空子集、非连续多句证据、超长输入零 forward 拒绝、测试 double 标记、模型 provenance、manifest 篡改、case-source 错绑、证据必须来自 retrieved Top-5、默认严格 reportable 评分路径，以及完整 300 条冻结包。Ruff check 与 format check 均通过。测试证明实现符合这些契约；质量数字仍只来自上述公开 dev 运行。

### SciFact v2 核心实现的源码与回归证据

| 实现 | 源码证据 | 回归证据 |
|---|---|---|
| 无标签句级适配 | [build_scifact_public_dev.py](../scripts/build_scifact_public_dev.py) 第 108、142–170、293–353、486–530 行：正文只含官方摘要句，身份/offset/hash 单独进入 sentence index，并固定 v2 schema | [adapter tests](../tests/test_scifact_public_dev_adapter.py) 第 111–190 行验证无 `[DOC]`/`[SENT]`/`Title:` 注入、正文覆盖和产物哈希 |
| Top-5 全子集判定与精确多句证据 | [scifact_judge.py](../src/evidencetrace/eval/scifact_judge.py) 第 199–214、260–283、295–360 行枚举最多 31 个非空子集、按原文顺序定位证据并统一评分 | [semantic Judge tests](../tests/test_scifact_semantic_judge.py) 第 197–258 行验证 31 个子集、rank 5 与 rank 2+4 非连续证据 |
| 真实模型门禁与无静默截断 | [run_scifact_public_dev.py](../scripts/run_scifact_public_dev.py) 第 445–567、869–899、911–963 行绑定 revision/权重/metadata，区分真实 scorer 与测试 double，并复用一次 Top-5 检索；[scifact_judge.py](../src/evidencetrace/eval/scifact_judge.py) 第 465–499、568–594 行在 forward 前检查完整输入且设置 `truncation=False` | [runner tests](../tests/test_scifact_public_runner.py) 第 328–440 行验证身份篡改拒绝及 300 cases/600 rows/8,195 pairs；[semantic Judge tests](../tests/test_scifact_semantic_judge.py) 第 462–524 行验证超限零 forward |
| 严格评分与证据绑定 | [score_scifact_public_dev.py](../scripts/score_scifact_public_dev.py) 第 134–189、352–493、527–934、1478–1579 行校验 pack/run/model/句级索引、证据属于 retrieved Top-5 且两路检索一致 | [scorer tests](../tests/test_scifact_public_scoring.py) 第 633–652 行使用正式 300-case 产物走默认严格评分；[verify_resume_metrics.py](../scripts/verify_resume_metrics.py) 独立重算关键简历指标与哈希 |

## 其他核心能力的源码与数据证据

### CLI、Controller 与五个 Agent

| 表述片段 | 源码证据 |
|---|---|
| Markdown/TXT CLI 主调用路径 | [src/evidencetrace/cli.py](../src/evidencetrace/cli.py) 第 57–172 行定义 check、完成预检并调用 SelfUseRunner；[src/evidencetrace/self_use_inputs.py](../src/evidencetrace/self_use_inputs.py) 第 51、158–164、455–534 行识别 .md/.markdown/.txt 并校验输入；[src/evidencetrace/self_use.py](../src/evidencetrace/self_use.py) 第 1784–1829 行解析 Markdown/TXT 后调用 ProductAuditPipeline。 |
| Python Controller 固定控制流 | [src/evidencetrace/product.py](../src/evidencetrace/product.py) 第 775–877 行创建角色并执行 mine、initial plan、review；第 1426–1489 行校验 Coordinator plan 并在失败时进入受限 fallback。Coordinator 可调用模型，所以不称整个 Controller 为“完全确定性”。 |
| 五个有界角色、两阶段计划与调用预算 | [src/evidencetrace/product.py](../src/evidencetrace/product.py) 第 304–310 行枚举五个角色，第 818–822 行实例化 Miner、Coordinator、Judge、Scout、Challenger，第 529–596、2064–2156 行维护 per-claim 状态和预算；[src/evidencetrace/agents/coordinator.py](../src/evidencetrace/agents/coordinator.py) 第 12–104 行限定 plan stage、action 和调用次数。数字“五”来自五个角色。 |
| 按 claim 隔离失败 | [src/evidencetrace/product.py](../src/evidencetrace/product.py) 第 529–544 行维护 per-claim 状态，第 1650–1678、2064–2156 行把 Scout、Judge、Challenger 或预算失败记录到当前 claim，而不是终止整个文档循环。 |

### 安全抓取

| 表述片段 | 源码证据 |
|---|---|
| HTTPX 实际请求路径 | [src/evidencetrace/retrieval/fetch.py](../src/evidencetrace/retrieval/fetch.py) 第 13、333–354 行导入 httpx、构造重试请求并实际调用 client.send，而不只是依赖清单声明。 |
| 内嵌凭据与敏感查询参数 | [src/evidencetrace/retrieval/fetch.py](../src/evidencetrace/retrieval/fetch.py) 第 42–63 行定义敏感 key；第 164–183 行拒绝 URL userinfo 和命中敏感 key 的 query。 |
| 私网、环回、link-local、metadata 地址 | [src/evidencetrace/retrieval/fetch.py](../src/evidencetrace/retrieval/fetch.py) 第 25–40 行定义 metadata host/IP；第 94–130、193–209 行使用 ipaddress 拒绝 private、loopback、link-local 等地址。 |
| DNS 全地址校验 | [src/evidencetrace/retrieval/fetch.py](../src/evidencetrace/retrieval/fetch.py) 第 135–220 行解析主机全部地址并逐个调用 IP 校验。 |
| 重试/重定向重新验证 | [src/evidencetrace/retrieval/fetch.py](../src/evidencetrace/retrieval/fetch.py) 第 285–345 行在跳转和每次请求 attempt 中重新进入 target/DNS 校验。 |
| 已验证 IP 建连 | [src/evidencetrace/retrieval/fetch.py](../src/evidencetrace/retrieval/fetch.py) 第 381–405 行把连接目标换成已验证 IP，同时保留 Host 与 SNI。 |
| 抓取边界 | [src/evidencetrace/retrieval/fetch.py](../src/evidencetrace/retrieval/fetch.py) 第 239–270、289–320、414–445 行限制超时、大小、重定向、重试、MIME 和流式读取。 |

“降低 SSRF/DNS 重绑定风险”是根据 URL/IP/DNS 校验与 IP pinning 源码作出的工程推断，不代表第三方安全认证或漏洞绝对不存在。

### JSON Mode 与严格判定契约

| 表述片段 | 源码证据 |
|---|---|
| 四个模型拥有字段 | [src/evidencetrace/audit_models.py](../src/evidencetrace/audit_models.py) 第 527–553 行的 LiveJudgeSemanticOutput 只包含 relation、confidence、reason、evidence_span，并要求实质关系提供 evidence。 |
| JSON Mode 与 Pydantic 严格校验 | [src/evidencetrace/audit_models.py](../src/evidencetrace/audit_models.py) 第 129–130 行禁止额外字段并冻结模型；[src/evidencetrace/model_client.py](../src/evidencetrace/model_client.py) 第 794–805、824–829、948–965 行把 schema 放入提示载荷、请求 JSON object，并在本地以 strict=True 校验。 |
| 身份字段本地装配 | [src/evidencetrace/agents/judge.py](../src/evidencetrace/agents/judge.py) 第 254–292 行从 JudgeInput 组装 canonical Verdict，而模型语义输出不拥有 claim/source/locator。 |
| 证据与 scope fail-closed | [src/evidencetrace/agents/judge.py](../src/evidencetrace/agents/judge.py) 第 313–347 行校验 claim/source/evidence、确定性冲突和缺失证据；[src/evidencetrace/models.py](../src/evidencetrace/models.py) 第 628–665 行约束 relation、confidence、source 和 evidence span。 |
| JSON、Markdown、SARIF 输出 | [src/evidencetrace/self_use.py](../src/evidencetrace/self_use.py) 第 1045–1067 行定义每个 target 的 audit.json、audit.md、results.sarif 路径，第 1540–1555 行把同一类型化 artifact 写成三种结果；[src/evidencetrace/sarif.py](../src/evidencetrace/sarif.py) 第 395–483、499–532 行构造、序列化并原子写入 SARIF 2.1.0。 |

self_use.py 当前仍是本地未跟踪文件；上述输出能力只能表述为当前本地工作树已经实现，不能写成已发布版本能力。在对外提供仓库前应先纳入版本控制并完成运行验证。

#### 补充实现（未写入简历）：确定性修复

| 表述片段 | 源码证据 |
|---|---|
| 五类标量及 fix 接线 | [src/evidencetrace/repair.py](../src/evidencetrace/repair.py) 第 148–155、1086–1170 行只识别 integer、decimal、percentage、date、semver；[src/evidencetrace/self_use.py](../src/evidencetrace/self_use.py) 第 1202–1321、1850–2013 行把 repair plan、人工确认和 apply 接入 fix 流程。数字“五”来自 ScalarKind 枚举。 |
| SHA-256、正反向 dry-run | [src/evidencetrace/repair.py](../src/evidencetrace/repair.py) 第 214–218、415–418、572–733、837、1009–1010、1471–1472 行校验摘要并验证 reverse 操作能恢复原字节。 |
| 原子替换 | [src/evidencetrace/repair.py](../src/evidencetrace/repair.py) 第 1680–1749 行在同目录创建临时文件、写入、fsync、二次校验目标并调用 os.replace。 |

repair.py、self_use.py、self_use_inputs.py 当前均是本地未跟踪文件；repair.py 第 1616、1649、1685 行还使用 O_NOFOLLOW/dir_fd 等 POSIX 接口，本次未在 Windows 上运行验证。因此该补充部分只能证明“当前本地工作树存在实现并已接入当前 CLI 源码”，不能写成已提交、已发布、跨平台验证或生产落地。

### 可复现评测体系

#### 数据集与原始输出

- [eval_sets/phase3g_dev_zh/dev.jsonl](../eval_sets/phase3g_dev_zh/dev.jsonl) 实际包含 72 行，每条记录的 provenance、split 与 annotation_status 标明 author-created、dev、provisional、not a holdout，且 document_path=null；[manifest.json](../eval_sets/phase3g_dev_zh/manifest.json) 第 2–16、37–41 行记录 case_count=72、六类 relation 各 12 例、independent_holdout=false、formal_gate_eligible=false、pair extraction not_applicable。
- [src/evidencetrace/eval/dataset.py](../src/evidencetrace/eval/dataset.py) 第 106–195 行实现 JSONL 加载、字段验证与数据集哈希，不是只在说明文档中声明数据集。
- 三个 run 的 [run_01 原始 JSONL](../eval_runs/phase3g_ownership_zh_dev_run_01/eval_results.jsonl)、[run_02 原始 JSONL](../eval_runs/phase3g_ownership_zh_dev_run_02/eval_results.jsonl)、[run_03 原始 JSONL](../eval_runs/phase3g_ownership_zh_dev_run_03/eval_results.jsonl) 各有 432 行，等于 6 个 baseline × 72 个 case；其中 Single Agent 和 Adaptive 每轮各 72 条。

#### Adaptive 基线确有源码

- [src/evidencetrace/eval/router.py](../src/evidencetrace/eval/router.py) 第 8–34、51–84 行根据 source availability、checkability、长度/分块信息选择 deterministic、full-context 或 retrieval-judge 路径，并给回退路由分配独立版本与 reason code。
- [src/evidencetrace/eval/baselines.py](../src/evidencetrace/eval/baselines.py) 第 489–582 行实际执行 adaptive_live，并记录 selected_route、reason 和调用上限。

#### fixed-six macro-F1 与 Token 汇总算法

- 六个标签由 [src/evidencetrace/models.py](../src/evidencetrace/models.py) 第 539–545 行的 Relation 枚举给出。
- [src/evidencetrace/eval/metrics.py](../src/evidencetrace/eval/metrics.py) 第 12、21–105 行按每个标签计算 TP/FP/FN、precision、recall、F1，再用六个标签的 F1 算术平均得到 fixed_taxonomy_macro_f1；第 371–421 行从原始 gold_relation/predicted_relation 生成这些指标。
- [src/evidencetrace/eval/metrics.py](../src/evidencetrace/eval/metrics.py) 第 191–219、247–256 行汇总输入 Token；[src/evidencetrace/eval/runner.py](../src/evidencetrace/eval/runner.py) 第 984–1014、1193–1234 行执行各 baseline 并写出逐条 JSONL 与 metrics.json。

### 量化行：评测执行量与产物完整性

- 三个 eval_results.jsonl 各有 432 行，来自 72 个 case × 6 个 baseline；三轮合计 1,296 条逐例记录。原始文件见 [run_01](../eval_runs/phase3g_ownership_zh_dev_run_01/eval_results.jsonl)、[run_02](../eval_runs/phase3g_ownership_zh_dev_run_02/eval_results.jsonl)、[run_03](../eval_runs/phase3g_ownership_zh_dev_run_03/eval_results.jsonl)。
- [artifact_manifest.json](../eval_runs/phase3g_ownership_zh_dev_stability/artifact_manifest.json) 的 artifact_count 为 20；本次逐文件重算 SHA-256 后为 20/20 匹配、0 missing、0 mismatch。
- 这组数字只证明评测执行规模、逐例留痕和本地文件完整性，不代表 1,296 个独立样本，也不证明模型泛化效果。

## 本轮 Phase4C 新冻结诊断复测

### 数据冻结、拆分与泄漏审计

- [数据 manifest](../eval_sets/phase4c_public_zh_diagnostic/manifest.json) 记录 **32 个 claim-source pair、4 个官方中文文档快照、dev/test 各 16 例**；JSONL 原始文件 SHA-256 为 `6c69799f59c1cfd8f48b74282f2c3da613c1edd4c4ac9ddb23496aa2a332d7e3`，[runner manifest](../eval_runs/phase4c_public_zh_diagnostic/runner_test/run_manifest.json) 和检索对照共同记录包含来源 fixture 的数据联合哈希 `21fde215fddb883c2eaf6e7efb31547e2b4ef15d7e66181dfdbd691bf4324fbe`。测试集含 4 entailed、3 partially_entailed、4 contradicted、3 not_in_source、2 not_checkable；其中前三类共 11 例拥有冻结的字面 gold evidence，可用于 literal retrieval 指标。
- [构建脚本](../scripts/build_phase4c_public_zh_diagnostic.py) 在预测与指标计算前固定来源、case 和分层配额；[leakage_audit.json](../eval_sets/phase4c_public_zh_diagnostic/leakage_audit.json) 对旧评测集检查 case ID、归一化 claim、来源 URL、来源内容哈希，四类 exact overlap 均为 0。审计不排除语义改写重叠或模型训练污染。
- manifest 明确标记 `benchmark_validity=diagnostic_contaminated`、`public_benchmark_eligible=false`、`annotation_review=none`。标签是由构建器按字面证据确定的诊断标签，没有独立人工复核，所以目录名中的 `holdout` 不能被包装成第三方公开盲测。

### CJK 检索与空召回退的源码证据

| 能力 | 源码证据 | 直接证明的事实 |
|---|---|---|
| CJK 基础检索修复 | [src/evidencetrace/retrieval/rank.py](../src/evidencetrace/retrieval/rank.py) 第 23–50、67–74、113–149、255–291 行 | `lexical-cjk-ngram-v3` 对中文做 NFKC 归一化并建立二/三元倒排；CJK 与 Latin/版本号候选取并集，排序前过滤非正分结果。 |
| 检索层有界回退 | [src/evidencetrace/retrieval/rank.py](../src/evidencetrace/retrieval/rank.py) 第 94–178、181–233 行；[src/evidencetrace/audit_models.py](../src/evidencetrace/audit_models.py) 第 48–49 行 | 多 chunk 回退必须由调用方显式启用，只接受可拼成连续原文且不超过 2,000 字符/8,000 UTF-8 bytes 的来源；返回 `score=0`，不伪造检索相关性分数。 |
| Adaptive 空召回路由 | [src/evidencetrace/eval/baselines.py](../src/evidencetrace/eval/baselines.py) 第 489–582 行；[src/evidencetrace/eval/router.py](../src/evidencetrace/eval/router.py) 第 8–34 行 | Adaptive 只检索一次；结果为空/全零且来源可用、估算上下文不超过 32,768 bytes 时，改走 `full_context_single_agent` 并记录 `empty_retrieval_full_context_fallback`，超预算仍保留 Retrieval+Judge 路径。 |
| Scout 接线与追踪 | [src/evidencetrace/agents/scout.py](../src/evidencetrace/agents/scout.py) 第 24、265–312 行 | Discovery 在无词法证据时调用同一有界全文回退，并输出 `retrieval_fallback` 与 `fallback_reason`，使回退可审计。 |

### 同一冻结数据上的 v1/v3 对照

[benchmark_phase4c_retrieval.py](../scripts/benchmark_phase4c_retrieval.py) 直接从 git commit `eca904ebdd77c78a838c18b24ba5be63e303e133` 读取 `lexical-cjk-ngram-v1` 源码作为基线，并与当前 `lexical-cjk-ngram-v3` 在相同 chunk、相同 query、相同 top-k=5 上对照；没有 checkout 或改写工作树。机器结果见 [retrieval_comparison.json](../eval_runs/phase4c_public_zh_diagnostic/retrieval_comparison.json)。

| 冻结子集 | N | v1 Recall@1 | v3 Recall@1 | v1 Recall@5 | v3 Recall@5 | v1 MRR | v3 MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| test substantive | 11 | 0.9091 | **1.0000** | 1.0000 | 1.0000 | 0.9394 | **1.0000** |
| test mixed CJK+Latin | 9 | 0.8889 | **1.0000** | 1.0000 | 1.0000 | 0.9259 | **1.0000** |
| all substantive（dev+test） | 19 | 0.8421 | **0.8947** | 1.0000 | 1.0000 | 0.9035 | **0.9386** |

- test substantive 中 v3 相对 v1 为 1 个 case 更好、10 个持平、0 个更差；Recall@1 差值 +0.0909 的 paired bootstrap 95% CI 为 **[0, 0.2727]**，MRR 差值 +0.0606 的 CI 为 **[0, 0.1818]**。区间均含 0，样本又只有 11 例，不能称为统计稳定提升。
- 全部 32 个可用 case 的空结果率由 v1 的 **0.03125（1/32）** 降至 v3 的 **0（0/32）**。这是检索是否返回候选的覆盖指标，不是关系判定准确率。
- 不使用本轮确定性 Judge 的 relation macro-F1 作为中文语义指标，因为其 token-F1 仍是 Latin/digit 口径；这里报告的是 gold evidence 字面片段的 Recall@k/MRR。

### 空召回探针、runner 状态与测试

- 对 4 个官方中文来源统一使用无匹配查询“量子纠缠熵”：默认检索 **4/4 为空**，显式回退后 **4/4 返回**；返回项 **4/4 为 `score=0`、4/4 是连续完整原文、4/4 满足 2,000 字符/8,000 bytes 上限**。该结果来自 [retrieval_comparison.json](../eval_runs/phase4c_public_zh_diagnostic/retrieval_comparison.json) 的 `empty_retrieval_policy_probe`，只验证回退不变量，不验证 Agent 语义质量。
- 标准 runner 在 test split 执行了 16 case × 3 个确定性 baseline，共形成 [48 条逐例记录](../eval_runs/phase4c_public_zh_diagnostic/runner_test/eval_results.jsonl)。[run_manifest.json](../eval_runs/phase4c_public_zh_diagnostic/runner_test/run_manifest.json) 显示 live 被请求，但 Single Agent、Retrieval+Judge、Adaptive 都因缺少 `OPENAI_API_KEY` 标记为 `skipped_missing_credentials`；[metrics.json](../eval_runs/phase4c_public_zh_diagnostic/runner_test/metrics.json) 的 `actual_model_calls=0`。
- 本轮聚焦测试命令覆盖 [Phase4C 冻结/泄漏/检索测试](../tests/test_phase4c_public_zh_diagnostic.py)、[空召回与 CJK 混合检索测试](../tests/test_empty_retrieval_fallback.py)、既有中文 Adaptive、抽取检索与产品纵切测试，实跑结果为 **71 passed**。测试证明本地实现满足断言，不替代真实模型对比或外部 benchmark。

## 冻结内部 holdout：为什么不能只挑正向数据

[execution_record.json](../eval_runs/phase3f_v3_zh_internal_gate/execution_record.json) 是一次性消耗的 72 例中文内部 holdout 记录，固定模型为 `deepseek-v4-flash`，无网络重抓取、Claim Miner 调用为 0。两条 live 路径的结果为：

| 指标 | Single Agent | Retrieval+Judge | 后者相对前者 |
|---|---:|---:|---:|
| fixed-six macro-F1 | **0.7274** | 0.6138 | **-0.1136** |
| contradiction recall | **1.0000** | 0.7692 | -0.2308 |
| evidence span F1 | **0.8906** | 0.7654 | -0.1252 |
| provider calls | 72 | 39 | -33 |
| total Token | **63,175** | 70,658 | **+11.84%** |
| model latency p95 | **5.218 s** | 11.690 s | **+124.05%** |

预注册门槛要求 macro-F1≥0.70、contradiction recall≥0.80；Retrieval+Judge 两项均未达到，`gate_passed=false`，所以正确工程动作是拒绝宣称“检索 Agent 优于 Single Agent”。它同时说明调用次数减少不等于 Token 或延迟下降：39 次调用使用了更长上下文，最终反而更贵、更慢。

## 内部开发评测结果（只可按限定口径引用）

以下数字用于保留研发过程和复现依据，只描述固定开发集上的 pair-level 判定结果，不作为完整系统效果或泛化能力声明。

### 从三轮原始 JSONL 独立复算

本次没有读取 aggregate.md 作为结论。复算直接遍历三个 eval_results.jsonl，分别按 single_agent_live 和 adaptive_live 过滤 72 条预测，再严格按上述源码的六标签公式计算：

可直接运行 [scripts/verify_resume_metrics.py](../scripts/verify_resume_metrics.py)，从原始 JSONL 重算 holdout/开发集指标、结构化输出恢复率与 20 份产物哈希；脚本不把 README 或本说明文档当作数据源。

| Run | 原始记录数 | Single Agent fixed-six F1 | Adaptive fixed-six F1 | Single 输入 Token | Adaptive 输入 Token |
|---:|---:|---:|---:|---:|---:|
| 1 | 432 | 0.8971980676328503 | 1.0000000000000000 | 48,905 | 30,720 |
| 2 | 432 | 0.9576719576719577 | 1.0000000000000000 | 48,905 | 30,720 |
| 3 | 432 | 0.9860869565217390 | 1.0000000000000000 | 48,905 | 32,128 |

原始值还可在三个机器生成的 metrics.json 中交叉核对：

- Run 1：[metrics.json](../eval_runs/phase3g_ownership_zh_dev_run_01/metrics.json) 的 baselines.single_agent_live.headline.verification.fixed_taxonomy_macro_f1、baselines.adaptive_live.headline.verification.fixed_taxonomy_macro_f1 与 live_execution.baseline_telemetry；
- Run 2：[metrics.json](../eval_runs/phase3g_ownership_zh_dev_run_02/metrics.json) 的相同 JSONPath；
- Run 3：[metrics.json](../eval_runs/phase3g_ownership_zh_dev_run_03/metrics.json) 的相同 JSONPath。

复算公式：

    Single Agent 平均 F1
      = (0.8971980676328503 + 0.9576719576719577 + 0.9860869565217390) ÷ 3
      = 0.946985660608849

    F1 绝对差值
      = 1.000000000000000 − 0.946985660608849
      = 0.053014339391150966
      ≈ 0.053014

    Adaptive 平均输入 Token
      = (30,720 + 30,720 + 32,128) ÷ 3
      = 31,189.333333333332

    Single Agent 平均输入 Token
      = 48,905

    输入 Token 降幅
      = 1 − 31,189.333333333332 ÷ 48,905
      = 0.3622465323927342
      ≈ 36.22%

三轮资源均值从各 run 的 `metrics.json` 中同口径汇总：

| 指标 | Single Agent | Adaptive | 改变量 |
|---|---:|---:|---:|
| fixed-six macro-F1 | 0.946986 | 1.000000 | +0.053014 |
| provider calls | 72.000 | 48.667 | **-32.41%** |
| input Token | 48,905 | 31,189.333 | **-36.22%** |
| total Token | 65,427.667 | 42,128.000 | **-35.61%** |
| pipeline p50 | 2.278 s | 1.991 s | **-12.59%** |
| pipeline p95 | 3.815 s | 3.760 s | -1.43% |

Adaptive 的路由构成为：12 例 `source_unavailable` 与 12 例 `not_checkable` 走零模型调用的确定性路径，17 例走全文判定，31 例走 Retrieval+Judge。这解释了为什么调用数与 Token 会下降；也解释了为什么该结果高度依赖开发集分层，不能直接外推生产分布。

### 结构化输出恢复指标

[aggregate.json](../eval_runs/phase3g_ownership_zh_dev_stability/aggregate.json) 的 `schema_recovery` 记录三轮共 **504 次逻辑模型调用**：首轮结构契约成功率 **0.992063**，4 次首轮 schema 失败均使用最多一次重试恢复，`unrecovered_schema_failures=0`。这个 99.21% 是“格式契约首轮成功率”，不是关系判定准确率。

### 原始产物完整性

[artifact_manifest.json](../eval_runs/phase3g_ownership_zh_dev_stability/artifact_manifest.json) 第 8–27 行保存三个 run 和 aggregate 机器产物的 SHA-256。本次重新计算其中 20 个文件摘要，结果为 20/20 匹配、0 mismatch；其中：

| 文件 | 重新计算的 SHA-256 |
|---|---|
| run_01/eval_results.jsonl | 23f4e432c327ab26815d67d843458c3712e308a52c1a10d49099d072675e64d6 |
| run_02/eval_results.jsonl | 2f4ea47994732ac6fdc8d99994c2f52bd6e2ef2b2a359b64d1eb0929237fe1fc |
| run_03/eval_results.jsonl | 5c3ec00ea20e5b5368ac3898f2d765d23be0a08d0a85a5c7debb53aa79a88d4b |
| aggregate.json | 708100ab584dec82124443f426974439b673c345037931cb98c0717d2149f9d9 |

哈希匹配只证明本地评测文件未偏离项目 manifest，不代表第三方复现或正式盲测通过。

## 如需引用评测必须保留的边界

- Phase4C 只能写成“32 例、4 个官方中文文档快照的冻结诊断集”；不能称第三方公开 benchmark、正式盲测或生产分布。其 11 例 test substantive 的 Recall@1/MRR 差值 CI 均包含 0，不能写成稳定提升。
- Phase4C 的 4/4 空召回结果是无匹配查询下的离线策略探针，不是 4 个真实业务问题、模型准确率或多 Agent 效果；本轮 live baseline 实际调用为 0。
- 必须写“冻结的 72 例中文 claim-source pair 开发集”；不能写测试集、盲测、生产数据或公开基准。
- pair 数据直接提供 atomic claim，manifest 明确 pair extraction not_applicable。因此该指标不测 Claim Miner，不能写成完整五 Agent 端到端 F1。
- 72 例的原始记录标注 author-created、provisional、not a holdout，并使用本地 fixture 来源；不能包装成第三方公开基准或真实用户数据。
- 1.000 是 fixed-six macro-F1，不是“生产准确率 100%”。
- 0.053014 是绝对 F1 差值，约 5.30 个百分点；不能偷换成相对提升 5.30%。
- 36.22% 只指平均输入 Token 相比 Single Agent 的降幅，不是总 Token、成本、延迟或吞吐改善。
- 当前本地工作树中 product.py、audit_models.py、models.py、sarif.py 等核心文件有未提交修改，repair.py、self_use.py、self_use_inputs.py 等关键文件尚未跟踪；这些能力不能表述为“已发布”“已上线”或“生产落地”。
