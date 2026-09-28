# EvidenceTrace 工程决策与取舍

本文用于回答面试中最常见的追问：为什么 EvidenceTrace 采用现在的架构，而不是更
简单或更流行的方案。

它不是新的设计规范。规范性权威是 Projects 根目录的
`evidencetrace_self_use_mvp_plan_2026-07-28.md` v2.0；下文所称“当前实现”只以
当前代码和 canonical artifacts 为准。历史评测数字仅是历史记录，不是新的验收结果。

## 1. 决策总览

| 决策 | 解决的问题 | 主要代价 |
|---|---|---|
| Agent 外设置 Deterministic Controller | 权限、预算、重试和状态不可由模型自批 | 编排代码更长 |
| 本地拥有 ID、locator 和 provenance | 模型会产生 schema-valid 的身份错配 | 本地组装与校验更复杂 |
| 模型前建立 source map | 行号幻觉和不可安全写回 | parser 成为关键基础设施 |
| Miner 使用 bounded windows | 长段落一次失败会吞掉全部 claims | 需要 coverage 与窗口状态模型 |
| citation/local 优先，discovery 受控 | 搜索成本、不可审计摘要和开放网络面 | 无证据时更容易得到 partial |
| FTS5 candidate filter、local BM25-style scorer 与 CJK n-gram | MVP 需要离线、可复现、无额外 key | 不等于通用语义检索 |
| safe fetch 后 exact span 才算证据 | snippet、SSRF 和网页噪声风险 | 不支持浏览器级动态网页 |
| strict typed contract，仅一次 schema recovery | 坏 JSON、无限 retry 和静默补值 | provider 不稳时会诚实失败 |
| per-claim 隔离并保留 partial | 单点失败不应污染或掩盖其他结果 | 状态不是简单成功/失败 |
| typed canonical state 驱动 terminal/Markdown/SARIF 与 decision state | 多视图状态分裂 | diff 仍由 RepairPlan/ApplyResult 生成 |
| deterministic scalar repair | 自由改写不可证明安全 | 只能修很窄的错误类型 |
| TTY human gate + 原子可逆写回 | 静默批准、stale 和半写风险 | `fix` 不能无人值守 |
| 冻结已消费评测，转向产品闭环 | 重复调参会制造虚假质量信心 | 无正式质量通过可宣传 |

## 2. 为什么 Agent 外还要有 Deterministic Controller

### 问题

如果让多个 Agent 直接互相发送自由文本并自行调用工具，就无法稳定回答：

- 谁可以访问网络；
- 谁决定 retry；
- 谁消耗了多少预算；
- 谁能修改文件；
- 一个 Agent 失败后，其他结果是否仍可信。

### 决策

五个 Agent 只拥有受限职责和 typed contract。Controller 独占工具、网络授权、
budgets、retry、policy、artifacts 和 side effects。

五个 Agent 是 Miner、Coordinator、Scout、Judge、Challenger。Controller、Router、
Retriever、Repair Planner 和 Writer 都不是 Agent。当前也不是五个进程、五个不同
模型或分布式自治系统；它们可以顺序复用同一个 OpenAI-compatible client。

### 为什么这样选

Multi-Agent 的价值在职责隔离，而不是数量。确定性控制面使权限和失败语义可以由
代码审查与测试，而不是依赖 prompt 中的“请不要越权”。

### 代价与未选方案

- 没有使用 LangChain/LangGraph。直接 Python 编排更适合当前小范围 MVP，也让
  budget、fallback 和 side effect 显式；代价是状态流转代码较长，扩展新流程需要
  手工维护。
- 没有开放式 Agent loop。Coordinator 只有 initial/review 两个有限阶段；这牺牲
  自主性，换取可终止性和可审计性。

### 代码入口

- `src/evidencetrace/product.py`
- `src/evidencetrace/self_use.py`
- `src/evidencetrace/agents/coordinator.py`

## 3. 为什么把语义所有权与身份所有权分开

### 问题

历史 Judge contract 要求模型回传 `claim_id`、`source_id`、locator 和版本字段。
模型可能给出语义上合理、schema 也合法、但 ID 与输入不一致的结果。此时无法知道
它判断的还是不是原 claim。

### 决策

模型只负责它真正需要推断的字段，例如 relation、confidence、reason 和
evidence span。本地代码从 trusted input 组装 claim/source identity、locator、
corroboration 和 contract version，并验证 evidence span 确实存在。

Miner 同样只让模型给出 claim text/type/checkability；本地 source map 验证 exact
character span，并组装 line、claim ID 和 citation binding。`AtomicClaim` 不持有
offset；repair 后续重新定位并换算 byte offset。

### 为什么这样选

这是最小权限原则：模型拥有语义，不拥有本地已经知道的身份和来源。这样可以减少
无意义的 echo 错误，也防止 schema-valid 的 provenance 漂移。

### 代价与边界

本地 assembly 和跨字段 validator 更复杂。它只能保证“判断对应正确输入”，不能保证
模型的语义判断一定正确。

### 代码入口

- `src/evidencetrace/audit_models.py`
- `src/evidencetrace/agents/judge.py`
- `src/evidencetrace/agents/miner.py`

## 4. 为什么在模型前建立可信 source map

### 问题

让模型返回行号或 offset 会产生位置幻觉；让模型自由改写 claim 又会破坏原文定位。
如果 audit、SARIF 和 repair 各自重新找文本，同一句重复出现时还可能绑定到不同位置。

### 决策

Markdown/TXT 先由 deterministic parser 生成模型可见文本和 trusted source map。
所有 claim 必须是可验证的 exact substring，并满足单调、不重叠和 protected-token
规则。后续 artifact、SARIF 和 repair 复用同一坐标。

### 为什么这样选

位置是文件系统事实，不是模型推理结果。先固定位置，才可能做准确提示、stale 检查和
字节级写回。

### 代价与未选方案

parser bug 会放大到整个下游，因此 inline code、citation、footnote、URL 和 Markdown
结构都需要专门测试。没有采用“让 LLM 返回大致行号再模糊搜索”的方案，因为重复文本
和最小修复无法安全处理。

### 代码入口

- `src/evidencetrace/markdown.py`
- `src/evidencetrace/local_evidence.py`

## 5. 为什么 Miner 从段落调用改成 bounded windows

### 问题

早期一个 paragraph 对应一次模型调用。长段落发生 length、empty content、schema
failure 或 protected-token omission 时，整段有效 claims 都会丢失。

### 决策

Controller 在调用模型前规划 source-mapped bounded windows；每个 window 独立
dispatch 和校验。claim validity 与 paragraph coverage 分离，未覆盖窗口通过
ET2001/`needs_human` 可见，并设置 window/attempt 硬上限。

### 为什么这样选

这是缩小故障爆炸半径的结构性修复。相比继续增加 token、prompt 或 retry，它能保留
其他窗口的有效结果，也更容易解释调用预算和遗漏位置。

### 代价与边界

窗口规划、exact monotonic mapping、protected occurrence 和 coverage policy 更
复杂。identifier coverage-only signal 只能暴露“合法空输出”，不能替模型创造
claim。当前 Miner P0 仍是 `improved_but_blocking`。

### 代码入口

- `src/evidencetrace/agents/miner.py`
- `src/evidencetrace/markdown.py`

## 6. 为什么 citation 和 local reference 优先

### 问题

对每条 claim 直接搜索网页会扩大网络面、成本和不确定性。搜索摘要可能看似相关，
但缺少稳定上下文，无法作为可审计证据。

### 决策

证据优先级是：

1. claim 自带 citation；
2. 用户显式提供的只读 local references；
3. 只有 `--discover` 授权时才允许 Scout/Tavily 查找候选 URL。

搜索响应可以携带 Tavily snippet，但当前 Controller 路径只消费候选 URL；snippet
不进入 evidence payload。候选 URL 仍必须经过 safe fetch、文本抽取和 exact
bounded-span 选择，才能送入 Judge。直接 citation 的 safe fetch 不要求
`--discover`。

### 为什么这样选

已有引用和本地规范通常最接近作者语境，也最容易复核。将 discovery 变成显式授权，
可以避免“有 Agent 就默认联网”的权限扩张。

### 代价与边界

没有合格证据时系统按路径保留 `source_unavailable`、`not_in_source`、
`needs_human` 或 partial，而不会让模型凭内部知识补全。`not_checkable` 来自 claim
本身的 checkability，例如意见或建议，不是“没找到证据”的通用标签。

### 代码入口

- `src/evidencetrace/product.py`
- `src/evidencetrace/agents/scout.py`
- `src/evidencetrace/local_evidence.py`

## 7. 为什么当前使用 lexical retrieval，而不是向量数据库

### 问题

MVP 需要同时处理 citation 页面和本地 references，但不应要求另一套 embedding
credential、远程索引或难以重现的相似度服务。

### 决策

当前用 SQLite FTS5 只筛选 Latin candidates，所有 candidates 都由本地 BM25-style
scorer 排序；无 FTS5 时扫描全部 chunks。中文路径直接扫描全部 chunks，并加入
NFKC CJK bigram/trigram 信号；数字、日期和版本信号继续保留。

### 为什么这样选

它依赖少、成本低、离线可运行、结果易复现，也足以支撑 bounded candidate routing。
对事实核验而言，精确术语、数字和版本往往是重要信号。

### 代价与未选方案

这不是通用语义检索，同义改写和跨语言表达会漏召回。当前没有 embeddings 或向量库；
若未来证据规模和语义召回需求扩大，应以独立 benchmark 决定是否引入，而不是把
MVP 的 lexical 结果包装成语义搜索。

### 代码入口

- `src/evidencetrace/retrieval/rank.py`
- `src/evidencetrace/local_evidence.py`

## 8. 为什么搜索结果还必须经过 safe fetch

### 问题

URL 获取会带来 SSRF、redirect、私网/metadata 访问、credential 泄露、超大响应、
非文本内容和网页噪声等风险。

### 决策

每次请求和 redirect 都重新验证 scheme、userinfo、敏感 query、DNS/IP、MIME、
大小和 timeout。HTML 使用受控 extractor 去除脚本与结构噪声，只输出 bounded exact
chunks。

### 为什么这样选

搜索服务只回答“可能去哪看”，不能回答“网页实际写了什么”。Judge 必须接收 fetched
exact span，才能在 artifact 中保留可复核 locator。

### 代价与未选方案

stdlib extractor 依赖少且容易测试，但不是浏览器级 reader；动态页面、登录页和
JavaScript 渲染不在当前范围。历史上也出现过 noise regex 误删正文，说明
deterministic extractor 同样需要真实页面回归。

### 代码入口

- `src/evidencetrace/retrieval/fetch.py`
- `src/evidencetrace/retrieval/extract.py`

## 9. 为什么 strict schema 只允许一次固定 recovery

### 问题

模型可能返回坏 JSON、缺字段或违反 strict schema。无限 retry、自动修 JSON、把失败
响应再次塞回 prompt，既扩大预算，也可能泄露 payload 或把错误“修成”表面成功。

### 决策

Agent contract 使用 Pydantic strict models。只有明确的 `ModelSchemaError` 可以用
固定 schema reminder 重试一次；transport、scope、budget 和本地 validation 失败
不重试。诊断只记录 payload-free 的 stage、field path、usage、长度和 truncation
signal。

### 为什么这样选

一次 recovery 可以处理偶发格式偏差，又不会形成调参循环。区分 schema error、模型
质量错误和系统完整性错误，也使 partial 的原因可审计。

### 代价与边界

provider contract 不稳时会产生 typed Agent failure，而不是尽力凑出结果；
Coordinator plan/schema failure 是特例，会记录 rejected plan 并走 deterministic
fallback。历史 full-document smoke 只证明一篇文档的 operational path，不证明
质量或 Multi-Agent 优势。

### 代码入口

- `src/evidencetrace/model_client.py`
- `src/evidencetrace/eval/full_document.py`

## 10. 为什么保留 per-claim partial，而不是全有或全无

### 问题

某一 citation 获取失败、某个 Miner window schema error 或单条 Challenger 失败，
不应抹掉同文档其他已验证结果；但也不能把整份文档显示成成功。

### 决策

失败按 window、claim 和 file 隔离，canonical artifact 同时记录局部 outcome 与
document status。共享初始化、输出/artifact 完整性或 revalidation 等 run-level
failure 可以中止；单 target pipeline failure 则尽量隔离为该 target fatal。

### 为什么这样选

事实审查天然允许“部分可核对”。显式 partial 比隐藏失败或放弃全部结果更适合人工
复核，也让恢复工作能定位到具体 claim。

### 代价与边界

状态模型更复杂，用户不能只看一个 exit code。当前最新 canonical run 正是
`document_status=partial`；这不是失败被隐藏，而是已知边界被保留。

### 代码入口

- `src/evidencetrace/self_use.py`
- `src/evidencetrace/audit_models.py`
- `src/evidencetrace/models.py`

## 11. 为什么 canonical audit 驱动其他输出

### 问题

如果 terminal、Markdown、SARIF 和 diff 各自在不同阶段读取不同对象，可能出现某条
claim 在 JSON 是 error、在终端却是 accepted 的状态分裂。

### 决策

每个 target 的 `audit.json` 是 canonical truth；typed artifact 对 claim、verdict、
resolution、Challenger、repair 和 apply 做跨字段约束。终端、`audit.md` 和 SARIF
从同一已验证状态派生。batch manifest 只做索引，不替代 per-target audit。

### 为什么这样选

一个 canonical state 能让人、CI 和恢复流程复核同一事实，也避免 renderer 反向修改
业务状态。

### 代价与边界

schema 演进必须同步所有 renderer。历史 legacy ArtifactManager 的“写后重读”不能
泛化为所有 self-use 路径都实际重新读取；当前准确说法是各视图来自同一 typed
canonical state。

### 代码入口

- `src/evidencetrace/self_use.py`
- `src/evidencetrace/render.py`
- `src/evidencetrace/sarif.py`

## 12. 为什么没有 Repairer Agent

### 问题

把 contradicted claim 直接交给另一个 Agent 重写，会引入措辞漂移、额外事实、单位
变化和不可预测的 edit span。即使 verdict 正确，修改也未必安全。

### 决策

deterministic repair planner 只处理有限 scalar：

- integer、decimal 和 percentage；
- 四种受限 date surface：`YYYY-MM-DD`、`YYYY/MM/DD`、`Month D, YYYY`、
  `D Month YYYY`；
- 受限 SemVer。

候选必须有 final contradiction、exact evidence、同类型/单位、可定位且无 overlap。
Challenger 必须 typed `uphold` contradiction，或在 `revise` 后仍由同一 exact
evidence 支持同一 replacement；`abstain`、error、incomplete 都不能修。替换值只能
从 bounded evidence-backed option 中选择，不能自由输入。

### 为什么这样选

Agent 做语义判断，确定性代码做字节修改。牺牲覆盖率，换取每个 edit 都可解释、可
预览、可逆。

### 代价与边界

普通文本、复杂表格、单位换算、`v` 前缀版本和任意 locale 日期不属于通用修复能力。
`candidate_patch=unverified` 仍需保留，不能因一次 acceptance 就宣称任意修复已验证。

### 代码入口

- `src/evidencetrace/repair.py`
- `src/evidencetrace/self_use.py`

## 13. 为什么 `fix` 必须交互、原子且可逆

### 问题

审计后到 apply 前，文件可能已被用户修改；路径父目录可能变化；多个 edits 可能重叠；
进程中断可能留下半个文件；用户也可能并未批准具体改动。

### 决策

- `fix` 必须运行在 interactive TTY，无 `--yes`；
- evidence trust、冲突 option 和最终 apply 分开确认；
- apply 前重新 pin 路径、读取文件并复核 SHA；
- 在内存验证 forward/reverse round trip；
- 先持久化 reverse diff；
- 同目录临时文件、`fsync`、`os.replace`；
- 每个文件独立处理，不承诺跨文件事务。

### 为什么这样选

用户批准的是具体 evidence 与具体 diff，不是笼统授权“让 Agent 修文档”。原子替换
消除单文件半写；reverse diff 提供明确恢复路径。

### 代价与边界

`fix` 不适合 CI 自动批准；多文件仍可能部分应用，但每个 target 的 outcome 会独立
记录。`check` 所称“只读”是不会修改 target/reference/Git index，它仍会写审计
artifacts。

### 代码入口

- `src/evidencetrace/cli.py`
- `src/evidencetrace/repair.py`
- `src/evidencetrace/self_use.py`

## 14. 为什么停止重复评测，转向自用闭环

### 问题

已消费 holdout 上继续调 prompt 或重跑会把开发适配误当成泛化能力。pair benchmark
也没有运行 Claim Miner，不能回答完整文档 extraction 质量。

### 决策

Phase 3 保持 `completed_with_known_limitations`，`phase4_eligible=false`；不启动新
v5，不重复消费 frozen holdout。工程工作转向多目标 check、local references、
canonical artifacts 和受证据约束的 interactive fix。

### 为什么这样选

指标不能替代可用产品闭环。诚实冻结不完整的质量结论，比继续在已知数据上取得更好
数字更可信。

### 代价与边界

项目目前不能宣称正式盲测通过、production-ready 或 Multi-Agent 优于 Single Agent。
这不是待用措辞包装的问题，而是必须保留的证据边界。

### 事实入口

- `docs/design-decisions.md`
- `docs/implementation_status.md`
- `docs/full-document-benchmark-design.md`
- `docs/known-limitations.md`

## 15. 当前未接入主链的模块

代码中存在 `CacheStore` 和 TOML config loader，但当前 self-use/product 主链没有调用
它们；CLI 当前也不能据此宣称 cache hit/TTL 或自定义 TOML 已生效。

面试时可以把它们描述为“已有辅助模块或历史基础设施”，不能列为当前运行能力。判断
某项能力是否真的存在，应从 CLI 到 product/self-use 调用链核对，而不是只看文件名。
