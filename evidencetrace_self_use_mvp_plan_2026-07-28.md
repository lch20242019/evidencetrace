# EvidenceTrace 自用多 Agent 文档审查与修正 CLI 计划

版本：1.0  
日期：2026-07-28  
状态：待合并到 Mac 权威计划并执行  
目标周期：1–2 个工作日

## 1. 项目重新定义

EvidenceTrace 的当前首要目标不是公开发布、获取 star 或完成完整评测，而是成为一个
开发者本人可以实际使用的多 Agent CLI：

> 对 AI 生成或人工编写的 Markdown、README、ADR、RFC 和 Git diff 提取事实
> claim，优先核查文档已有引用；在用户授权时受控联网搜索补充证据；区分已支持、
> 矛盾、未验证和不可核查内容，并只为有充分证据的修改生成候选 diff。

“修正”必须理解为 evidence-grounded suggestion，而不是让模型自由重写文档：

- 只生成独立 diff，绝不自动 apply、stage 或 commit。
- 证据不足时返回 `needs_human`，不得猜测修正。
- 联网不可用时返回 `discovery_unavailable`，不得伪装为事实错误。
- Partial 是正常、安全的产品结果，不等于完整事实审计。

## 2. 为什么这是多 Agent 项目

保留现有受控架构，不再重新设计 Agent：

1. **Miner**：从原文提取连续、逐字符一致的 atomic claims、citation 和位置。
2. **Coordinator**：每篇文档最多两次 typed plan，无网络、文件或 shell 权限。
3. **Scout**：仅在用户授权且证据不足时执行有界 query、候选 URL 和 safe fetch。
4. **Judge**：每次只判断一个 claim 和受限 evidence。
5. **Challenger**：只复核矛盾、低置信度、数字、日期、版本和比较类 claim。
6. **Deterministic Controller**：拥有预算、权限、重试、并发、工具执行、artifact、
   cache、policy、exit code 和 Agent plan 拒绝权。

用户可观察的 Agent trace 必须证明这不是一个单 prompt 包装项目：

- 显示实际调用和跳过的 Agent。
- 显示跳过或 fallback 原因。
- 显示预算消耗和剩余预算。
- 单 claim/window 失败不影响其他 claim。

## 3. 当前可复用基线

以下能力已经存在，本计划不重新实现：

- Markdown/README/ADR/Git diff 解析和 changed-paragraph 选择。
- DeepSeek/OpenAI-compatible live model 路径。
- source-mapped bounded Miner windows。
- strict schema、exact substring、protected-token guard 和有限 recovery。
- Coordinator、Scout、Judge、Challenger 和 deterministic Controller。
- claim/window failure isolation。
- canonical `audit.json`、`audit.md`、terminal 和 SARIF。
- ET2001 paragraph mining warning。
- coverage-only inline-code identifier signal。
- optional `--suggest-patch`，只生成不应用。
- 无 key、无网络的 deterministic demo。

当前已知限制继续保留：

- Miner 仍可能漏掉核心 claim。
- 真实 Tavily HTTP discovery 尚未完成验证。
- Patch 安全边界已验证，但内容质量只做最小自用验收。
- Miner P0 继续为 `improved_but_blocking`。
- Phase 3 继续为 `completed_with_known_limitations`。
- `phase4_eligible=false`。

## 4. 本轮唯一用户流程

目标命令保持现有公开 CLI，不新增第二套入口：

```bash
uv run evidencetrace check draft.md \
  --model deepseek-v4-flash \
  --discover \
  --sarif outputs/audit.sarif \
  --suggest-patch outputs/suggested.diff
```

命令必须产生：

1. Terminal summary。
2. Canonical `audit.json`。
3. Human-readable `audit.md`。
4. SARIF。
5. 可选 `suggested.diff`。
6. Agent 调用、跳过原因和预算记录。

输入文件和 Git index 在运行前后必须完全一致。

## 5. 1–2 天内必须完成的工作

### 5.1 自用 UX 收口

- 给出一条可复制命令和最少环境变量说明。
- `--help` 必须能解释 citation-first、`--discover` 和 patch 不自动应用。
- Terminal 必须清楚区分：
  - supported/entailed；
  - contradiction；
  - `needs_human`；
  - `discovery_unavailable`；
  - not-checkable。
- `needs_human`/`discovery_unavailable` 不得伪装为
  `not_in_source` 或已证伪关系。
- 失败时给出可操作原因，不输出原始模型 response 或 credential。

### 5.2 文档证据路径

- 有 citation 的 claim 优先使用 citation，不调用 Scout。
- Citation URL 必须经过现有 safe-fetch 边界。
- 引用不足且用户未传 `--discover` 时，安全返回人工处理。
- 不得用搜索 snippet 直接作为 Judge evidence。

### 5.3 联网搜索路径

- `--discover` 只授权现有 Scout，不允许人工搜索冒充 Agent。
- 保持最多 2 query、5 candidate URL、3 safe fetch 的既有边界。
- 使用现有 Tavily adapter；本轮不增加第二搜索 provider。
- 缺少 `TAVILY_API_KEY` 时明确返回 `discovery_unavailable`。
- 若必须验收真实联网搜索，执行前需要用户提供 Tavily key；只有 DeepSeek key
  不能替代搜索 provider。
- Key 只检查 configured/missing，绝不打印、保存或散列其值。

### 5.4 候选修正

只为满足以下条件的 claim 生成候选 diff：

- claim 已定位到可信 source span；
- Judge/Policy 已形成可支持的 contradiction 或明确替换依据；
- replacement 有受限 evidence；
- 修改保持最小、可定位和可审阅。

不得：

- 因为 `needs_human`、`discovery_unavailable` 或 not-checkable 自动改写；
- 接受 rejected/withheld Miner draft；
- 让 LLM 重写整段或整篇；
- 自动运行 `git apply`；
- 自动 stage、commit 或 push。

候选 diff 必须通过 `git apply --check` 或等价的只读语法检查，但不得真正应用。

## 6. 自用验收案例

只创建一个小型、可理解的自用案例，不建立 benchmark：

- 输入是一份 AI 生成的技术 Markdown。
- 基于一个有稳定公开文档的项目或协议。
- 包含四类内容：
  1. 有直接 citation 且可支持的 claim；
  2. 无 citation、但允许 Scout 后可发现证据的 claim；
  3. 一个明确的数字、日期或版本矛盾；
  4. 一个 opinion 或无法外部核查的句子。

开发阶段使用 deterministic mocks 和合成 fixture。代码冻结后只进行一次最终
真实自用运行，不根据最终结果继续调 prompt 或挑选输出。

若没有 Tavily key：

- citation-first 路径仍必须完成；
- discovery 路径必须安全显示 `discovery_unavailable`；
- 不得声称真实联网搜索已经验收；
- 真实 Scout 验收保留为唯一外部 prerequisite。

## 7. Definition of Done

以下条件全部满足，才可称为“自用多 Agent CLI 完成”：

1. 新环境中能够安装并运行当前 CLI。
2. 离线 demo 无 key、无网络稳定完成。
3. 自用 Markdown 能被 Miner 拆成 source-faithful claims。
4. Citation-backed claim 能进入 Judge。
5. Discovery 未授权或缺 key 时安全降级。
6. 如配置 Tavily，至少一个 unresolved claim 完成真实
   Scout → safe fetch → Judge 路径。
7. 数字/版本矛盾能形成明确 finding。
8. 不可核查内容不会被强行判真或判假。
9. 至少生成一个非空、可定位、可通过语法检查的候选 diff；若证据不足则宁可为空。
10. `audit.json`、`audit.md`、terminal、SARIF 和 diff 语义一致。
11. 每个 Agent 的调用或跳过原因可观察。
12. 输入 SHA、working-tree input diff 和 Git index 前后不变。
13. Credential/privacy scan 为 0 命中。

## 8. 最小验证策略

不再为每个步骤运行全部测试。

开发中只运行本次修改直接相关的 focused tests：

- operational status rendering；
- discovery unavailable；
- citation-first routing；
- candidate patch generation/non-application；
- Agent trace 和预算。

完成后统一运行一次：

- focused suite；
-现有 boundary-compliant suite，继续排除已记录的 7 个 consumed loaders；
- changed-production mypy；
- changed-file Ruff；
- compileall；
- `git diff --check`；
- privacy/credential scan；
- 离线 demo 双跑；
- 最终自用 CLI acceptance。

测试数量不是进度指标，不创建新 benchmark、holdout 或 eval phase。

## 9. 两日执行安排

### Day 1

1. 冻结现有 Miner/window 架构。
2. 核对完整 `check --discover --sarif --suggest-patch` 路径。
3. 修正仍存在的未验证状态表达问题。
4. 确认 patch 只生成、不应用，并补最少 focused tests。
5. 准备自用 Markdown 和可重复命令。

### Day 2

1. 完成 focused validation。
2. 在代码冻结后执行一次最终自用运行。
3. 检查 Agent trace、evidence、SARIF 和候选 diff。
4. 做一次完整的最终回归、隐私和副作用检查。
5. 写一个简短的本地使用说明和真实限制。
6. 形成可审阅 checkpoint；commit/push 仍需用户单独确认。

## 10. 明确不做

- 不继续针对旧三个案例优化 Miner。
- 不修改 bounded-window policy。
- 不扩大 protected-token 或 candidate-span 架构。
- 不做完整 provider telemetry。
- 不做 Web UI、PDF、数据库或服务端部署。
- 不做 GitHub 包装、star 优化、宣传视频或 PyPI。
- 不做 10 案例 dogfood。
- 不声称 recall、accuracy、F1 或 production-ready。
- 不让“写进简历”取代真实可运行和可审计证据。

## 11. 简历价值

完成后，这个项目可以真实证明：

- 设计了受 deterministic Controller 约束的多 Agent 工作流，而不是开放式 Agent loop。
- 使用 typed contracts、预算、有限 retry 和 failure isolation 控制 LLM 风险。
- 构建了 citation-first、可选安全搜索、单 claim Judge 和高风险 Challenger。
- 将 Agent 结果产品化为 CLI、canonical audit、SARIF 和不可自动应用的候选 diff。
- 对真实文档进行了 dogfood，并诚实保留 partial 和 known limitations。

建议简历描述：

> Built a bounded multi-agent CLI that audits and proposes evidence-grounded
> corrections for AI-generated technical Markdown, using citation-first
> verification, controlled web discovery, typed agent plans, failure
> isolation, SARIF output, and unapplied patch generation.

## 12. 计划责任

- **原计划依据：** EvidenceTrace v3.0 要求从评测工程转向真实用户闭环，并保持
  citation-first、受控 Agent、失败隔离和 patch 不自动应用。
- **用户最新目标：** 应聘 Agent 工程/后端岗位，重点证明多 Agent 和工程质量；
  最低要求是本人可实际使用，不要求演示或立即公开 GitHub。
- **判断：** 停止发布包装和继续扩展 Miner，将唯一目标收敛为一个可自用的
  evidence-grounded Markdown 审查与候选修正 CLI。
- **计划修改：** 建议在 Mac canonical plan 追加
  `[2026-07-28 Self-Use Multi-Agent CLI Scope]`，并明确本计划取代
  Portfolio Alpha/GitHub-ready 优先级，但不改写历史状态。

## [2026-07-28 Self-Use Multi-Agent CLI Execution]

### 计划依据

本轮只收口现有公开 `check` 路径，修复阻断本人使用的 CLI 说明、operational
状态表达和候选 patch 证据门槛。Miner、bounded windows、prompt、token、retry、
Agent 数量、Controller 预算、Judge 和 Policy 均保持冻结；不扩展评测、发布或
额外本地参考文档。

### 实际修改

- `check --help` 现在明确 citation-first、`--discover` 是显式 Scout 授权、
  缺少搜索 key 时返回 `discovery_unavailable`，以及 patch 不会 apply、stage
  或 commit。
- Canonical `audit.json` 新增向后兼容的 `claim_operational_outcomes`。只有
  completed claim 才进入事实 `verdicts`；`needs_human`、
  `discovery_unavailable`、Agent error 和 budget exhaustion 由独立 operational
  outcome 表达。Terminal、`audit.md` 和 SARIF 从 canonical audit 渲染同一
  ET2002 human-review finding，不伪装成 `not_in_source`。
- `source_unavailable` 继续保留为合法的来源可用性 relation。
- 候选 patch 只接受带受限 evidence span 的 completed
  `contradicted` verdict；其他 relation 和 operational outcome 不再生成 patch。
  Patch 仍只原子写入独立文件，从不自动应用。
- 本机安装了 `uv 0.11.32` 以验证计划中的实际 `uv run` 入口；没有修改项目依赖
  或发布配置。

### 离线验证

- Focused suite：`81 passed`。
- Boundary-compliant suite：`604 passed`，继续排除既有 7 个 consumed-corpus
  loader。
- Changed-production mypy、scoped Ruff、compileall 和
  `git diff --check` 通过。
- 无 key、无网络的 `uv run --offline evidencetrace demo` 双跑逐字节一致，
  SHA-256 为
  `3b0e01bce663795c4b31787dac47ccc00a92e1b30278beb405a21354d39c5ee4`。

### 一次最终自用运行

冻结输入为 `examples/self-use/draft.md`，包含 6 个 AI 风格技术陈述和一个稳定的
Python 官方 release-page citation；它不是 benchmark。输入 SHA-256 为
`0afbaec8f3cbd7372fd435d9c4081439bbd4860bbe690111f3018cb0485d7987`。
公开 CLI 只运行一次，退出码为 2，document status 为 `partial`；运行后输入和
空 Git index 的 SHA-256 均保持不变。

实际 Agent 路径和预算：

- Miner：9 windows planned/dispatched、9 provider attempts，保留 5 个 exact
  claims；4 个 paragraph 记录
  `miner_scope_missing_protected_token`，未静默丢失。
- Coordinator：2/2 typed planning calls。
- Judge：3 次，三个 cited claims 均进入 citation-first Judge。
- Scout：4 次 Agent dispatch。三个以安全 `scout_error` 结束；一个 uncited
  claim 明确形成 `discovery_unavailable`。
- Challenger：1 次高风险复核。
- Controller 最终记录 search queries 2/6、safe fetches 3/20。

进程环境中没有配置 `TAVILY_API_KEY`，因此 Tavily HTTP 请求为 0；没有人工搜索
冒充 Scout，真实联网 discovery 尚未验收。DeepSeek logical Agent dispatch
合计 19；除 Miner 的 9 次 provider attempts 外，当前安全 artifact 不保存其他
Agent 的 schema-recovery attempt 总数，因此真实 provider attempts 总数为
`not_available`，不作推测。

### 自用结果与已知限制

- 五个保留 claim 均为原文 exact substring。
- 三个 cited claims 均到达 Judge，但 citation source 没有形成候选 evidence
  span；唯一 completed verdict 为 `not_in_source`，预期的日期 contradiction
  未形成。
- 一个 uncited claim 正确显示 `needs_human/discovery_unavailable`；另一个
  Scout planning path 为 `scout_error`。
- Opinion 所在 paragraph 没有被强判真或假，而是以 ET2001
  `needs_human` 暴露；但没有形成显式 `not_checkable` verdict。
- `audit.json`、`audit.md`、terminal 和 SARIF 的 operational 状态、文件、行号
  和 reason 一致。SARIF 包含 4 个 ET2001、4 个 ET2002 和 1 个 ET1002。
- 因没有 completed contradiction 和受限 replacement evidence，
  `suggested.diff` 安全保持空文件，SHA-256 为标准空文件哈希；未 apply。

### Definition of Done 判定

整体 **未达到** Definition of Done。已满足：`uv run` CLI、离线 demo、
source-faithful claims、cited Judge handoff、缺 key 安全降级、Agent trace/预算、
四种输出一致、输入/index 不变和隐私边界。未满足：真实 Tavily Scout、清晰
contradiction、显式 not-checkable verdict、非空 evidence-grounded patch。

下一项唯一阻塞是 citation evidence retrieval/handoff 未能从已成功抓取的稳定
官方页面形成可供 Judge 使用的 span，因而无法完成 contradiction 和非空 patch
验收。本记录不预授权修复、第二次 live run、P1、发布或范围扩张。

状态继续保持：

- Miner P0 = `improved_but_blocking`
- Phase 3 = `completed_with_known_limitations`
- `phase4_eligible=false`

## [2026-07-28 Self-Use Citation/Discovery Closure — Authorized Adjustment]

### 授权依据与边界

- 原计划中的“一次最终真实运行”已由 run
  `20260728T051838Z-034290e8` 消耗；该运行暴露了 citation source 已抓取、但
  没有形成 Judge 可用 evidence span 的自用阻塞。
- 当前 CLI 进程已确认 `TAVILY_API_KEY` 为 configured；只检查配置状态，不打印、
  保存或散列 credential。用户要求在现有受控产品路径中完成真实
  Scout -> Tavily -> safe fetch -> Judge 验证。
- 本调整只额外授权一次代码冻结后的 remediation live recheck，输入继续固定为
  `examples/self-use/draft.md`，模型继续固定为 `deepseek-v4-flash`。
- 在 live recheck 前只允许离线判断 citation evidence 是否在进入 Judge 前
  确定性丢失。只有根因为 deterministic handoff bug 时，才允许修改
  fetched-content extraction、bounded candidate selection、evidence binding
  或 Judge request assembly 中直接导致丢失的位置及其聚焦测试。
- 禁止修改 Miner、window/coverage/protected-token、任何 Agent prompt 或
  contract、模型参数、预算、retry、Tavily adapter、patch eligibility policy
  和自用验收输入。
- 禁止挑选结果、连续调参、人工 retry 或重复运行。额外 live recheck 无论成功
  或失败都只执行一次并停止。

### Citation 根因与最小修复

- 冻结 run 保存的 citation content SHA-256 为
  `7de100b5c95cd4a9f661e9867571c9261908c91d0a73fbc5304881619d32d3fa`。
  诊断性 safe fetch 得到逐字节相同的内容；原始 HTML 含所需发布日期和版本
  文本，但现有 extractor 产生 0 chunks。
- 根因分类为 deterministic handoff bug。HTML noise 属性正则把属性值中的任意
  `nav` 子串都当作导航区域；页面的外层 wrapper 因而使整页正文在 retrieval 和
  Judge 之前被跳过。
- 最小修复只把 noise attribute 改为 token-boundary 匹配。`nav` 标签和
  `header`、`footer`、`cookie` 等既有噪声仍被排除；safe-fetch、lexical
  ranking、Judge/Scout contract、prompt、模型、预算、retry 和 patch
  eligibility 均未改变。
- 同一冻结正文在修复后形成 32 个 bounded chunks，三个 cited claims 各有 5 个
  lexical candidates，且候选 evidence 是其 bounded chunk 的真实 substring。
  通用 MockTransport 回归测试确认 fetched content 进入真实
  `claim_judgement` request schema，导航文本不进入 evidence。

### 离线冻结验证与新环境

- Focused suite：`71 passed`。
- Boundary-compliant suite：`605 passed`，继续排除既有 7 个 consumed-corpus
  loaders。
- Changed-file Ruff、changed-production mypy、compileall 和
  `git diff --check` 通过；任务 diff credential scan 为 0。
- Offline demo 双跑逐字节一致，SHA-256 仍为
  `3b0e01bce663795c4b31787dac47ccc00a92e1b30278beb405a21354d39c5ee4`。
- 独立 `/tmp` 虚拟环境完成依赖同步，`evidencetrace --help`、
  `evidencetrace demo` 和 `evidencetrace check --help` 均成功。仓库开始时没有
  已有 `uv.lock`；uv 解析生成的临时 lock 仅保留于 `/tmp`，未作为项目修改
  留在工作区，因此“使用既有 lock”不适用。

### 唯一一次 remediation live recheck

- 冻结后仅执行一次原命令，run ID 为
  `20260728T065626Z-40dc78f9`。退出码为 2，document status 为 `partial`；
  输入 SHA-256 和空 Git index 均保持不变，没有 retry 或第二次运行。
- Miner 预先规划并 dispatch 9 个 windows，记录 9 次 provider attempts；
  保留 6 个 source-faithful claims。三个 cited claims 均越过已修复的
  extraction/retrieval handoff 到达 Judge，不再记录 `evidence_not_found`。
- Coordinator 使用 2/2 calls。最终 trace 为 Scout 2 次、Judge 4 次、
  Challenger 4 次。c1/c2 在 Challenger 阶段以安全 `challenger_error` 隔离，
  c3 为 `judge_error`，c4 为 `scout_error`；没有保存或推测原始模型输出。
- Tavily live path 已验证。c5 经 Coordinator 授权进入 Scout，执行 2 个受约束
  query；Tavily 返回的非 citation candidate 至少一个经过 safe fetch，形成
  source `https://www.python.org` 并进入 Judge。总 fetch 计数为 6，其中 3 次
  是 citation fetch、3 次是 Scout candidate fetch。Search snippet 没有进入
  evidence assembly。
- c5 的最终 relation 为 `not_in_source`。c6 的 opinion 被明确表示为
  `not_checkable`，没有被强判 entailed 或 contradicted。
- Canonical contradiction 未形成：c1/c2 的高风险 review 在 Challenger 阶段
  失败，c3 的 Judge 失败。没有 completed contradiction 和受限 replacement
  evidence，因此 `suggested.diff` 安全保持空文件；candidate patch 尚未验证，
  也没有执行 `git apply` 或实际应用。
- Terminal、`audit.json`、`audit.md` 和 SARIF 一致。SARIF 包含 3 个 ET2001、
  4 个 ET2002、1 个 ET1002 和 1 个 ET1005。

### 最终状态

- `self_use_cli = partial`
- `Tavily live path = verified`
- `citation handoff = repaired`
- `candidate patch = unverified`

Definition of Done 仍未全部达到。已补齐真实 Tavily Scout、citation evidence
handoff、显式 not-checkable 和新环境 smoke；剩余阻塞是 downstream
Judge/Challenger 可靠性没有形成 canonical 日期 contradiction，因而也没有非空、
evidence-grounded candidate patch。本调整不预授权继续修复或再次 live run。

历史状态继续保持：

- Miner P0 = `improved_but_blocking`
- Phase 3 = `completed_with_known_limitations`
- `phase4_eligible=false`
