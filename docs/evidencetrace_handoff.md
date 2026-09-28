# EvidenceTrace 新窗口交接文档

更新日期：2026-07-25

## 1. 新窗口首先要知道的事情

最新项目不在当前服务器副本中，而在用户的 Mac：

```text
/Users/USERNAME/Projects/evidencetrace
```

真人评审记录位于仓库外：

```text
/Users/USERNAME/Projects/evidencetrace_review_records
```

当前服务器的 `/data/yz_data/evidencetrace` 是旧副本，不能用它继续最新开发。
新窗口必须以 Mac 上的仓库为准，并在修改前检查 `git status` 和 `git log -1`。

用户希望使用中文交流，偏好简洁、直接、解释清楚原因。若无法准确理解意图，
应先提问，不要自行扩大项目范围。

## 2. 项目最终目标

EvidenceTrace 的最终目标不是评测平台，而是一个真正可使用的受控 Multi-Agent
技术文档事实审计工具：

```text
EvidenceTrace CI: a factual linter for AI-written technical Markdown.
```

目标用户把 Markdown、ADR、RFC、README 或 Git diff 交给系统后，系统应自主完成：

```text
Markdown / Git diff
-> Claim Miner
-> Audit Coordinator
-> citation retrieval 或 Evidence Scout
-> Claim Judge
-> 条件式 Challenger
-> Policy
-> audit.json / audit.md / terminal / SARIF / PR annotations
```

成功标准是用户能够在五分钟内运行工具，并在真实文档或 Pull Request 中看到
可定位、可复核的 claim-source 审计结果。

评测只是验证手段，不能再成为项目主线。

## 3. 已确认的产品决策

用户已经确认以下范围：

1. 核心产品是审查已有 Markdown、ADR、RFC、README 和 PR，不是自动生成研究报告。
2. 缺少或不足引用时，Evidence Scout 可以主动联网寻找官方或高可信来源。
3. 使用 3C 混合架构：
   - Deterministic Controller 掌握预算、权限、状态、重试和最终执行权。
   - Audit Coordinator Agent 根据文档和 claim 状态动态调度预定义子 Agent。
   - Coordinator 不能创建任意 Agent、工具、system prompt 或无限循环。
4. Challenger 只处理矛盾、低置信度、数字、日期、版本、比较等高风险 claim。
5. 单个 claim 失败后继续处理其他 claim，最终输出 partial result。
6. 可以生成建议 patch，但必须由用户确认，禁止自动修改或提交。
7. v0.1 包括 CLI、GitHub Action、SARIF、Quickstart、Demo repo 和 PyPI/uvx。
8. 发布前应成功审查 10 篇真实文档并跑通一个真实 PR；不再要求新 blind holdout。
9. 保留旧评测 artifact 和必要回归测试，但停止扩张评测系统。
10. 保持一个 OpenAI-compatible ModelClient，首个真实验证模型是
    `deepseek-v4-flash`。

## 4. 3C 架构的准确含义

这不是自由生成子 Agent 的主 Agent，也不是完全固定的无 Agent 流水线。

```text
Deterministic Controller
├── 管理预算、权限、状态、重试、工具和 artifact
└── Audit Coordinator Agent
    ├── 选择 verify_citation
    ├── 选择 discover / Evidence Scout
    ├── 选择 Claim Judge
    ├── 选择 Challenger
    └── 选择 request_human
```

Coordinator 只能输出严格的 typed plan。Controller 必须验证：

- claim ID 是否真实存在；
- 是否重复分配任务；
- action 是否属于允许枚举；
- 是否超过搜索、模型调用和 token 预算；
- 是否试图授予新工具或创建未注册 Agent。

Coordinator 可以按需激活预定义子 Agent，但不能突破 Controller 的权限边界。

## 5. 当前已实现能力

当前项目已经实现：

- Markdown parser、citation 解析和行号映射；
- Git diff 和 changed paragraph；
- Pydantic typed contracts；
- artifact manager、suppression 和 policy；
- safe HTTP/HTTPS fetch；
- SSRF、DNS、redirect、timeout、MIME 和大小保护；
- HTML evidence extraction；
- SQLite FTS5、BM25/lexical retrieval 和 fallback；
- 中文 NFKC CJK bigram/trigram retrieval；
- 单 chunk invariant 和 Adaptive Router；
- Claim Miner Agent；
- Claim Judge Agent；
- 数字、日期、版本、否定、比较和实体 deterministic checks；
- evidence 必须为 source 的真实连续 substring；
- OpenAI-compatible ModelClient；
- deterministic fake model；
- `audit.json`、`audit.md` 和 terminal renderer；
- SARIF 2.1.0；
- GitHub workflow；
- schema-only 单次 recovery；
- 模型字段和本地可信字段的 ownership 隔离；
- full-document provisional benchmark 和相关指标实现。

Router、Retrieval、Policy 和 Renderer 是确定性组件，不应称为 Agent。

## 6. 当前尚未完成

以下是主要产品缺口：

- Audit Coordinator Agent 尚未实现；
- Evidence Scout 尚未实现；
- Challenger 尚未实现；
- 没有真正可用的 Web search adapter；
- 产品 pipeline 尚未完整支持 per-claim failure isolation；
- 没有完成 3 至 10 篇真实文档 dogfood；
- 没有公开 Demo repo；
- 没有完成 PyPI/uvx 发布；
- 没有在真实 PR 中验证完整体验；
- full-document 真实质量仍未完成评估；
- 没有 Web UI、PDF、向量数据库、第二 provider，这些也不属于当前范围。

## 7. 重要评测历史

项目曾过度投入 Phase 3 formal evaluation。必须保留历史事实，但不要继续这条路线。

### v3_zh

- Single Agent macro-F1：0.727360
- Retrieval-to-Judge macro-F1：0.613757
- Retrieval-to-Judge contradiction recall：0.769231
- v3_zh 已 consumed，禁止重跑。

### v4_zh

- 正式 gate 在第三个 `single_agent_live` case 因本地 unknown validation failure
  fail-closed。
- 没有生成正式质量指标。
- Phase 3 最终状态：`completed_with_known_limitations`
- `phase4_eligible=false`
- v4_zh 已 consumed，禁止重跑，不创建 v5。

### Phase 4A

已完成 SARIF、GitHub workflow 和 post-schema typed error hardening。

Commit：

```text
1c5628b6cb1ca42517bc59f0c3acd6e6894cf40b
```

### Phase 4B

已建立 full-document provisional benchmark：

- 24 篇 Markdown；
- dev/test 各 12；
- 120 个 author-constructed provisional gold claims；
- 72 个本地 source fixtures；
- fake model 完整运行仅证明契约和指标实现，不证明真实质量。

Commit：

```text
033f32dc89aa04afa95dd793f6dafcb82fe5746b
```

### Phase 4B 初次 live smoke

首个 `single_agent_document_live` 两次 schema failure 后停止。

Commit：

```text
69f26f49e8a2a14ad75e63e16928fa6a95c3972f
```

### Phase 4B.1 最新状态

最新本地 HEAD：

```text
a0051dd62c5f4123a0e5a99929a00927b24be41d
```

已完成：

- full-document `max_tokens=8192`；
- reason 最长 240 字符；
- 安全 envelope/JSON/Pydantic diagnostics；
- provenance、预算和 cache key 更新；
- 生产代码净增 356 行。

真实 smoke：

```text
Single Agent:
  2 provider attempts
  第一次 schema failure: claims/0/value_error
  finish_reason=stop
  非截断
  唯一一次 schema retry 成功

Multi-Agent:
  Miner 4 calls，全部首次成功
  Judge 5 calls，全部首次成功
```

这说明真正的 Miner -> Judge 路径已在一篇 dev 文档上 operationally 成功。
Single Agent 只是对照基线，不应继续阻塞产品开发。

工作区在该 checkpoint 后为 clean，没有 remote，没有 push。

## 8. 环境变量

现有模型配置：

```text
OPENAI_API_KEY
OPENAI_BASE_URL
EVIDENCETRACE_MODEL=deepseek-v4-flash
```

不得在聊天、日志或 artifact 中打印 API key。

未来 Scout 计划增加：

```text
TAVILY_API_KEY
```

Tavily 只用于返回候选 URL。搜索 snippet 不能直接作为 evidence；URL 必须经过
EvidenceTrace 现有 safe fetch、正文提取、retrieval 和 substring 校验。

## 9. 从现在开始禁止继续的事情

- 不创建 v5 或任何新 pair-level holdout；
- 不要求用户再次完成真人标注；
- 不重跑 consumed v2、v3、v3_zh、v4_zh；
- 不把 full-document Single Agent baseline 当作产品阻塞项；
- 不新增评测 phase、freeze bundle 或 reviewer provenance 流程；
- 不为了简历添加无必要 Agent；
- 不实现 PDF、向量数据库、Web UI、第二 provider 或自动报告生成；
- 不允许 Agent 无限重试；
- 不自动修改或提交用户文档；
- 不宣称已有正式 blind-holdout F1；
- 不宣称当前已经证明完整 Multi-Agent 优于 Single Agent。

## 10. 下一项权威任务

下一步不是完整 dev benchmark，而是实现 v0.1 受控 Multi-Agent vertical slice。

必须完成：

1. 找到并将 canonical `multi_agent_researchops_project_plan.md` 更新为 v3.0。
2. 保留旧阶段历史，但把主路线改为：
   - Agent vertical slice；
   - 真实文档 dogfood；
   - Quickstart、Demo repo 和发布。
3. 实现 Audit Coordinator Agent：
   - 每篇文档最多 initial plan 和 review plan 两次调用；
   - 输出 typed `ExecutionPlan`；
   - 只能调度预定义 action；
   - Controller 校验后才执行；
   - Coordinator 失败时 deterministic fallback。
4. 实现 Evidence Scout：
   - constrained query；
   - 查询不得增加原 claim 中不存在的实体、数字、日期或版本；
   - 每 claim 最多 2 个 query、5 个候选 URL、3 次 safe fetch；
   - 小型 `SearchClient` protocol；
   - 直接使用 `httpx` 的 Tavily adapter；
   - 离线 `FixtureSearchClient`；
   - 缺少 key 时返回 `discovery_unavailable`，不能崩溃。
5. 实现 Challenger：
   - 只处理高风险或 Coordinator 指定的 claim；
   - 只能输出 `uphold / revise / abstain`；
   - 每 claim 最多一次；
   - counter-search 必须交给 Scout。
6. 实现产品模式的 per-claim failure isolation：
   - Claim 状态：completed、agent_error、source_error、budget_exhausted、
     needs_human；
   - Document 状态：complete、partial、failed；
   - 单 claim 失败后继续；
   - evaluation runner 可以继续 fail-closed。
7. 完善 CLI：

```bash
evidencetrace check <path>
evidencetrace check <path> --discover
evidencetrace check <path> --suggest-patch <output.diff>
evidencetrace demo
```

8. `--suggest-patch` 只能生成候选 patch，禁止自动 apply 或 commit。
9. Demo 必须完全离线，并能展示四个 Agent 的条件触发和 trace。
10. 本任务不调用真实模型或 Tavily。

## 11. 下一项任务的工程约束

- 优先复用现有 parser、fetch、retrieval、ModelClient、Judge、policy、SARIF；
- 不自研通用 Agent SDK；
- 不复制一套平行 pipeline；
- 生产代码净增目标 900 至 1,400 行；
- 超过 1,500 行前必须停止并缩减设计；
- 使用 focused integration tests；
- 证明有引用、无引用、高风险、claim failure、Coordinator 越权、
  query provenance、snippet 非 evidence、offline demo 和 patch 不自动应用；
- 运行允许的测试、compileall、Ruff、diff check 和 privacy scan；
- consumed 数据只做 byte-level SHA-256；
- 创建本地 commit：

```text
v0.1-bounded-multi-agent-vertical-slice
```

- 不添加 remote，不 push；
- 完成后停止，等待真实文档 dogfood 授权。

## 12. 产品完成路线

完成 vertical slice 后，只剩三类工作：

1. 真实 dogfood：
   - 先 3 篇，再扩展到 10 篇真实技术 Markdown；
   - 修复真实用户阻塞问题；
   - 不再创建人工 holdout。
2. 发布体验：
   - 三到五分钟 Quickstart；
   - 一个真实 GitHub PR/SARIF 示例；
   - Demo repo；
   - PyPI/uvx；
   - security、limitations 和 architecture 文档。
3. 轻量质量报告：
   - extraction、relation、端到端成功率；
   - Agent 调用、token、延迟和失败率；
   - 明确标记为真实文档 dogfood，不冒充公开 benchmark。

## 13. 新窗口的沟通要求

- 不要继续用“再建一个 holdout”解决问题；
- 不要因为 Single Agent baseline 失败而阻塞主产品；
- 每次建议新增模块时，先说明它对真实用户流程的直接价值；
- 区分 product resilience 和 evaluation fail-closed；
- 对不确定的用户意图先提问；
- 汇报重点应是用户能完成什么，而不是新增多少 hash、manifest 或 phase。

