# EvidenceTrace 简历相关关键问题与修复记录

**当前状态：候选契约修复已实现，但本地实测质量退化，默认不启用；Judge 质量未解决。**

## 2026-09-05：生产 Judge 的提示未完整说明既有引用契约

### 真实问题

冻结 SciFact train OOF 证据的 30 题本地诊断产生 22 次主调用，其中 15 次
未通过生产校验：8 次 schema 错误、6 次引用越界、1 次确定性冲突。加上
3 次独立合成探针，总计 25 次调用均以 `stop` 结束，最长生成 192/512
Token；本轮不能把失败归因于生成额度截断。

证据：[原始诊断报告](../eval_runs/scifact_local_production_judge_train_pilot_v1/report.json)、
[原始生成日志](../eval_runs/scifact_local_production_judge_train_pilot_v1/generation_journal.jsonl)。
这些文件保持不变。

既有 `LiveJudgeSemanticOutput` 运行时验证要求实质关系提供非空引用，生产
Judge 又要求引用是单个候选证据片段中的连续原文。然而传入模型的 JSON Schema
只把 `evidence_span` 声明为字符串或 null，没有表达这些跨字段和原文范围约束。
失败中确实有 null 引用、换行被替换为空格、跨片段拼接和复制 claim/示例。

这不是全部问题的解释：无关证据错判支持等语义错误也已观察到。固定正向示例
是否诱发类别偏置仍是待验证假设，不能将它写成已确定根因。

### v5 候选：最小契约说明修复

仅在 `OpenAICompatibleClient._output_contract` 针对
`LiveJudgeSemanticOutput` 追加五条说明：

- `entailed`、`partially_entailed`、`contradicted` 必须提供非空引用。
- 非 null 引用必须是单个 `input.evidence` 文本的连续精确子串，保留空白和换行，不能拼接片段。
- 从证据原文复制引用，不从 claim 或格式示例复制。
- 真正的 `not_in_source` 仍允许 null，不能为了引用而强行回答。
- reason 非空且最多 240 字符，沿用原上限。

保持原输出示例、JSON Schema、系统提示、模型配置、检索与选择阈值、调用预算、
生成参数和运行时校验强度不变；未增加重试、格式修补、JSON grammar 或云端调用。
其他输出 schema 不添加这些 Judge 专属规则。

`MODEL_PROVIDER_CONTRACT_VERSION` 从 v4 更新为 v5，因为该版本已经是
`cache_key` 的组成部分。版本更新防止旧提示结果被新提示缓存命中；没有删除
旧缓存或旧实验文件。除该必要的出处/缓存版本外，不引入其他实验变量。

源码：[请求契约](../src/evidencetrace/model_client.py)、
[既有 schema](../src/evidencetrace/audit_models.py)、
[既有 Judge 校验](../src/evidencetrace/agents/judge.py)、
[缓存键](../src/evidencetrace/cache.py)。

### v5 实现时的离线验证

新增离线测试：[test_judge_prompt_contract.py](../tests/test_judge_prompt_contract.py)，
覆盖实际请求中的专属规则、原输入/示例/schema/预算不变、其他 schema 不受影响、
实质关系缺引用仍失败、NEI/null 仍合法、240 字符边界和 v4/v5 缓存键分离。
v5 实现时新增 13 项用例；与模型缓存、冻结证据 Judge、离线 worker、结构化输出契约
回归合并执行，结果为 **100 passed、3 deselected**。排除的是会接触历史 dev
或历史失败产物的用例；未执行模型前向、GPU 初始化或外部 API。Ruff lint 和
格式检查均通过。离线测试只证明请求与验证契约，不证明 LLM 遵守规则。

相同测试入口随后用于下述 v6 模式隔离回归。实际定向命令（项目根目录）为：

```powershell
& '..\env\evidencetrace-scifact\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests/test_judge_prompt_contract.py tests/test_model_cache.py tests/test_frozen_production_judge.py tests/test_scifact_local_judge_worker.py tests/test_structured_output_contract.py -k 'not runner_failure_artifact_contains_only_safe_schema_diagnostics and not historical'
& '..\env\evidencetrace-scifact\Scripts\python.exe' -B -m ruff check --no-cache src/evidencetrace/model_client.py tests/test_judge_prompt_contract.py
& '..\env\evidencetrace-scifact\Scripts\python.exe' -B -m ruff format --check --no-cache src/evidencetrace/model_client.py tests/test_judge_prompt_contract.py
```

终端或产物中的“执行完成”、退出码 0，以及
`completed_diagnostic_not_a_quality_acceptance`，只表示诊断过程完成，不代表
模型质量已通过验收，也不是可直接写入简历的效果提升。

### v5 真实同样本复测：契约失败减少，但质量退化

复测已完成，不再是“等待执行”。新产物为
[scifact_local_production_judge_train_contract_v2](../eval_runs/scifact_local_production_judge_train_contract_v2/report.json)，
官方评分为 [official_metrics.json](../eval_runs/scifact_local_production_judge_train_contract_v2_official_score/official_metrics.json)。
本轮与旧轮均使用同一 30 条 train claim、同一冻结 OOF 证据、3 个独立合成探针、
同一 Qwen 模型与环境、相同调用与生成上限。25 个实际请求的模型可见差异仅为
新增五条说明；没有改 gold、检索或运行时校验。新旧结果均保留。

| 指标 | 原 v4 提示 | 候选 v5 提示 |
| --- | ---: | ---: |
| 主调用数 | 22 | 22 |
| 主调用校验失败 | 15 | 12 |
| schema / 引用越界 / guard 失败 | 8 / 6 / 1 | 6 / 6 / 0 |
| Abstract Rationalized F1 | 0.193548 | 0.058824 |
| Sentence Label F1 | 0.113208 | 0.037736 |
| 11 个完整证据上下文中有效且标签正确 | 3 | 3 |
| 实际输出同时满足正确标签与完整证据的文档 | 3 | 1 |
| 有非空候选的 NEI claim 正确拒答 | 0/2 | 0/2 |
| 主调用实际输入 Token | 13,209 | 16,201 |
| 主调用实际输出 Token | 2,049 | 2,476 |

v5 主调用的六个 schema 失败分别为四个 reason 过长、一个 null 实质引用、一个
非法 JSON；六个范围失败包括大小写改写、换行变空格和复制 claim。两个原先
格式有效且判断正确的合成探针新增了代码围栏，因而被生产 JSON 校验拒绝；无关
证据被判支持的第三个探针仍未纠正。25 次生成均正常 stop，最长 189/512 Token。

原始主返回标签由 SUPPORT/CONTRADICT 的 12/10 变为 2/20，模型明显偏向反驳。
例如 claim 804 将原来的正确支持改为错误反驳；claim 853 虽标签正确却只引用句 8，
未覆盖 gold rationale 的句 7 和 8；claim 400 标签正确却引用非 gold 句。
“完整上下文中三个标签正确”不代表原文引用质量保持。

因此，补充正确规则并没有自动改善当前模型表现；未发现本轮接线或解析代码改错，
但这不能证明某条新增说明、正向示例偏置或模型大小是唯一根因。该结果仅为重复
train 诊断，不是新 holdout 或多 Agent 成绩。不能通过放宽校验、将错误算作 NEI、
只宣传 15→12 来掩盖质量下降。

### v6：隔离实验开关，默认恢复原请求

基于上述负结果，不把 v5 宣布为生产默认优化。当前客户端新增显式
`experimental_judge_scope=False` 参数：

- 默认 False：模型可见请求恢复为原 v4 的内容，不追加候选五条规则。
- 显式 True：追加完全相同的 v5 候选规则，保留用于受控诊断，不继续改提示。
- 两种模式均保留全部既有 schema、原文范围和确定性校验，NEI/null 规则不变。
- 契约版本更新为 v6；默认 `prompt_version` 为
  `openai-compatible-provider-contract-v6`，候选为该值加
  `-experimental-judge-scope`，防止把两种模式当作同一提示版本。

客户端自身不读写模型响应缓存。`cache_key` 已包含显式传入的 `prompt_version`；
未来使用缓存的调用方必须传入实际 `client.prompt_version`。当前实验仅在显式
诊断入口使用，不新增生产 CLI/UI 自动开关。未删除旧缓存，未改变旧评测目录。

已先原样归档刚跑过的 v5 客户端源：
[model_client.provider-contract-v5.py](archive/model_client.provider-contract-v5.py)，
SHA256 为 `8f2726555a8393c5904545f1a8f6bbe138efbcfc4d533944f565faca0ac29ae5`，
与 v2 运行预注册的源绑定完全一致。后续源代码改变不会被伪装成该轮实际源。

模式隔离回归扩充至 34 项用例，加入默认/候选请求与两轮保存请求完全一致、非法
布尔参数拒绝、两模式及历史 v4/v5 缓存键互异等验证。相关五个测试模块合并结果为
**121 passed、3 deselected**。此次只做离线测试，没有再次运行模型或调用 API。

**候选默认禁用；Judge 的科学关系判断、正确拒答和完整引用质量仍未解决。**
后续如继续修改，必须重新明确单一改动、冻结输入与预算、同时验收错误类型和质量，
不能用代码测试通过代替真实模型效果。

### 最终入口与交叉核验

- 诊断 runner 新增默认关闭的 `--experimental-judge-scope`；预注册明确记录模式，只有显式传参才启用候选。原 30 个 ID 的采样算法及 SCHEMA 保持不变。两个 CLI 回归验证默认与显式模式的转发，不加载模型。
- 原 v2 实际 runner 源码已保留为 [run_scifact_local_production_judge.contract_v2.py](archive/run_scifact_local_production_judge.contract_v2.py)，SHA256 为 `bf26dd57482348f0da511d47a74a3e642aec40704ac443ef06b128623ab60128`，与实际运行绑定一致。归档仅供追溯，不作为新入口。
- 新旧 30 条 claim/contexts、3 条独立探针、上游冻结哈希、模型及生成预算一致；实际模型可见差异只有五条候选规则。v2 主进程与 worker 正常退出，worker exit_code=0。一次预检的 WinError 5 来自沙箱目录写权限；允许创建新项目内目录后预检成功，不是原生内存崩溃。
- 官方评分前使用[行序对齐](../eval_runs/scifact_local_production_judge_train_contract_v2_order_alignment/manifest.json)，30 个 ID 的标签与引用内容不变。官方、独立实现及 runner 的四组共 12 项指标一致。
- 旧 v1 的 13 项输出、新 v2 的 13 项输出和新官方评分的 4 项输出均再次核验 manifest 哈希一致，未覆盖旧结果。
- 最终六组聚焦回归为 **127 passed、3 deselected**；四个本轮源码/测试文件的 Ruff lint 和 format 检查通过。排除项仍为历史 dev/失败产物用例；历史请求回放在缺少本地归档时明确 skip，固定 synthetic 请求断言仍执行。

最终回归命令（项目根目录，无模型调用）：

```powershell
& '..\env\evidencetrace-scifact\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests/test_judge_prompt_contract.py tests/test_model_cache.py tests/test_frozen_production_judge.py tests/test_scifact_local_judge_worker.py tests/test_structured_output_contract.py tests/test_scifact_local_production_judge_runner.py -k 'not runner_failure_artifact_contains_only_safe_schema_diagnostics and not historical'
```

## ET-DOC：简历指标来源及范围纠正

状态：已修改 [resume-project-evidence.md](resume-project-evidence.md) 的正文与最新状态；历史机器数据保持原样。

- 改为“项目简介—技术处理—公开验证结果”，不擅自填写核心开发者身份、实习经历或起止时间。
- 不再把 11 例诊断、72 例自建开发集或历史结构合规率作为正文质量亮点。
- 保留公开 SciFact train OOF 的证据选择结果：完整选中至少一组 gold rationale 的 claim 比例 208/505→265/505。它不是句级 Recall、完整生产链路效果或多 Agent 增益。
- 234/304=76.97% 是 selector 的无证据正确空选率，不是生产 Judge 的拒答准确率；双阈值尚位于校准/评测模块，不声称生产已部署。
- 本轮 30 题来自公开 train，但已被用于诊断，不是新 holdout。3 个合成探针只定位错误、与主指标分账，不能称为公开数据或用作简历准确率。

## ET-OPEN：尚未解决且限制当前项目定位的事项

- **完整文档抽取：**留存的完整文档运行仍有 Miner `partial/needs_human`；失败隔离和告警不等于漏抽已补回。SciFact claim-pair 测试绕过 Miner，仍需完整文档验收。[原始运行](../.evidencetrace/runs/20260728T065626Z-40dc78f9/product_run.json)
- **多角色收益：**固定证据的公开 train 本地角色消融未观察到复核增益，不能把单阈值/双阈值的选句成果转写成多 Agent 提升。[对照产物](../eval_runs/scifact_frozen_evidence_role_train_v1/comparison.json)
- **运行模式：**部分 CLI 入口在缺少 API Key 时将 model 设为 None，可能与用户指定模型的预期不一致；尚未修复，应明确区分 demo/live 并对 live 配置缺失报错。[CLI](../src/evidencetrace/cli.py)

本轮没有完成新的独立 holdout、完整文档/服务端到端验收或付费模型对照。因此不能把两个提示/格式修复概括成“系统所有关键问题已解决”。
