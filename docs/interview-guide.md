# EvidenceTrace 面试前必读指南

本文是 EvidenceTrace 的面试入口，面向 Agent 工程和后端工程岗位。它负责回答：

1. 项目解决什么问题；
2. 为什么需要 Multi-Agent 与 Deterministic Controller；
3. 每条技术链路如何工作；
4. 哪些结果已经验证，哪些仍是限制；
5. 面试时如何准确、可复核地表达。

本文不是新的产品计划。规范性权威仍是 Projects 根目录的
`evidencetrace_self_use_mvp_plan_2026-07-28.md` v2.0；实现事实以当前代码和
canonical artifacts 为准。

## 1. 建议阅读顺序

面试前至少按顺序读完：

1. 本文：建立项目全貌和讲解顺序；
2. [组件深挖](component-deep-dive.md)：掌握每部分的作用、手段、原因和代码入口；
3. [工程决策与取舍](engineering-decisions.md)：准备“为什么这样设计”的追问；
4. [真实问题与复盘](problems-and-lessons.md)：准备故障、诊断、修复和反思；
5. [已知局限](known-limitations.md)：明确不能夸大的内容。

需要现场打开代码时，再查看 [架构与安全边界](architecture.md) 和
[当前交接快照](PROJECT_HANDOFF.md)。

## 2. 一句话项目介绍

EvidenceTrace 是一个受 Deterministic Controller 约束的 Multi-Agent
Markdown/TXT 事实审查与交互式修复 CLI：

> 它从一个或多个技术文档中提取 source-located claims，优先核对 citation 和本地
> references，必要时受控联网检索，再由 Judge 判定，并对高风险 verdict 最多进行
> 一次 conditional Challenger 复核；对于有 exact evidence 的数字、日期和版本错误，
> 只在交互式 TTY 中逐条确认后原子写回，并生成可逆 diff。

## 3. 它解决的真实问题

技术文档经常出现以下风险：

- 版本、日期、比例或配置事实已经过期；
- 文档有 citation，但 citation 内容并不支持原 claim；
- 无 citation 的 claim 需要本地规范或网页证据；
- 一次模型失败会导致整份审计中断；
- 模型输出行号、source ID 或 evidence locator，难以信任；
- 自动修正文档时容易越界、覆盖新修改或留下不可恢复的部分写入；
- CI、终端、Markdown 和 SARIF 可能从不同内存状态渲染，产生互相矛盾的结果。

EvidenceTrace 的重点不是让模型“猜得更准”，而是把 claim、证据、Agent 权限、失败
状态、artifact 和文件写回全部放入可检查的工程边界。

## 4. 用户流程

只读审计：

```bash
uv run evidencetrace check TARGET1.md TARGET2.txt \
  --reference REF1.md --reference REF2.txt --discover
```

交互式修复：

```bash
uv run evidencetrace fix TARGET1.md TARGET2.txt \
  --reference REF1.md --reference REF2.txt --discover
```

`check` 永远不修改 target、reference 或 Git index，但会写审计 artifacts。`fix`
也不是自动改写器：它先复用相同审计链路，只有候选通过 evidence、Judge、
Challenger、scalar type/unit、位置和 overlap 等规划检查，才会询问 human 是否
信任/选择证据以及是否应用；批准后、真正写回前才重新 pin 路径并复核
target/reference SHA。

## 5. 系统主链

```text
explicit targets + shared references
        │
        ▼
preflight / canonical paths / safe IDs / SHA
        │
        ▼
source-mapped parser
        │
        ▼
Miner ──► local-reference lookup ──► Coordinator initial plan
                                      │
                                      ├── citation safe fetch
                                      ├── attached local candidates
                                      └── authorized Scout/Tavily + safe fetch
        │
        ▼
evidence routing / optional human trust or conflict selection
        │
        ▼
Judge ──► Coordinator review ──► conditional Challenger
        │
        ▼
canonical audit / policy / terminal / Markdown / SARIF
        │
        └── fix only: deterministic scalar repair
                    → apply confirmation
                    → stale check + atomic write
                    → suggested/applied/reverse diff
```

五个 Agent 是 Miner、Coordinator、Scout、Judge 和 Challenger。

Controller、parser、retriever、repair planner 和 artifact renderer 都是确定性组件，
不能为了“多 Agent”叙事把它们算成 Agent。

## 6. 为什么采用 Multi-Agent

采用 Multi-Agent 不是因为 Agent 数量越多越好，也没有证明 Multi-Agent 质量优于
Single Agent。选择它是为了职责和上下文隔离：

- Miner 只看目标文本并提取 claim，不看 evidence；
- Scout 只做受控 URL discovery，不下最终结论；
- Judge 一次只比较一个 claim 与 bounded evidence；
- Challenger 只对高风险 verdict 做一次独立复核；
- Coordinator 只输出 typed plan，没有文件、网络或写回权限。

权限、预算、重试、工具和写回由确定性 Controller 统一拥有。这样即使某个 Agent
失败，也能形成明确的 per-claim partial 状态，而不是让另一个 Agent 自主补全或
隐藏失败。

## 7. 技术栈及用途

| 技术 | 用途 | 选择原因 |
|---|---|---|
| Python 3.11+ | CLI、模型契约、文件与网络控制 | 适合类型化数据处理和快速构建本地工具 |
| Typer | `check`、`fix`、`demo`、`eval` CLI | 参数声明清晰，支持交互 TTY |
| Pydantic v2 strict models | Agent 输入输出和 canonical artifacts | 把 schema failure 与语义 verdict 分开 |
| markdown-it-py | Markdown tokenization | 保留结构，同时由本地代码维护可信 source map |
| httpx | citation/Tavily URL 的 HTTP 请求 | 支持显式 timeout、redirect 和安全验证 |
| SQLite FTS5 candidate filter + local BM25-style scorer | 每个 source/reference 内的 lexical retrieval | 无需 embedding key，MVP 可离线、可复现 |
| CJK 2/3-gram | 中文 lexical fallback | 在不引入分词服务的前提下保留中文召回信号 |
| OpenAI-compatible API | Miner/Judge 等 model-backed Agent | 与具体 provider 解耦，仍使用严格本地 schema |
| Tavily（可选） | Scout discovery | 只在 `--discover` 授权后使用，snippet 不作证据 |
| SARIF 2.1.0 | CI/code-scanning 集成 | 让 source-located findings 进入标准工具链 |
| `uv` + `uv.lock` | 可复现开发环境 | 新电脑不复制不可移植的 `.venv` |

## 8. 当前状态怎么解释

当前状态不能压缩成一个“完成”：

```text
self_use_check=operational_partial
latest_run_document_status=partial
citation_handoff=repaired
tavily_live_path=verified
not_checkable_path=verified
interactive_fix=accepted
candidate_patch=unverified
overall_new_self_use_goal=incomplete
Miner P0=improved_but_blocking
Phase 3=completed_with_known_limitations
phase4_eligible=false
```

正确解释：

- 交互式 `fix` 的窄范围 acceptance 已完成；
- 最新历史 `check` run 仍是 partial，并保留 downstream Agent errors；
- Miner 架构改善了失败隔离，但核心 claim coverage 没有完全解决；
- 历史评测没有形成正式质量通过；
- 项目已证明窄范围临时副本 interactive-fix acceptance 及其安全边界；overall goal
  和质量结论仍 incomplete，不能证明 production-ready 或质量优势。

## 9. 30 秒讲法

> EvidenceTrace 是我做的一个 bounded Multi-Agent 文档事实审查 CLI。它把 claim
> extraction、证据 discovery、verdict 和 challenge 拆给五个职责受限的 Agent，
> 但把权限、预算、网络、artifact 和写回保留在确定性 Controller。它支持 citation、
> 本地 reference 和受控 Tavily；`check` 永远只读，`fix` 只对 exact-evidence
> scalar correction 做逐条确认、原子写回和可逆 diff。项目最有价值的部分不是 Agent
> 数量，而是 source mapping、失败隔离、typed contracts 和安全写回。

## 10. 两分钟讲法

可以按“问题—架构—安全—结果—局限”展开：

1. **问题：** 技术文档里的版本、日期和配置会过期，普通 LLM 容易丢失 source
   location、引用错误或直接改写原文。
2. **架构：** parser 先建立可信 source map；Miner 提取 exact claim；Controller
   按 citation、本地 reference、授权 discovery 路由 evidence；Judge 判定；
   Challenger 对高风险结果复核。
3. **安全：** Agent 不拥有文件或网络权限；snippet 不是 evidence；schema 只允许
   一次 recovery；错误按 claim 隔离；同一已验证 typed artifact state 派生
   JSON、Markdown 和 SARIF。
4. **修复：** repair planner 不是 Agent，只处理证据中已有的同类型 scalar；
   human 逐条确认后，执行 stale check、atomic replace 和 reverse diff。
5. **结果与局限：** 真实 citation/Tavily 和交互式临时副本 fix 路径已验证，但
   最新历史 check 仍是 partial，Phase 3 和 Miner P0 限制继续保留。

## 11. 五分钟讲法

推荐先按时间控制结构，再选择三条技术主线：

```text
0:00–0:30  用户问题和一句话目标
0:30–1:15  端到端流程与五 Agent/Controller 边界
1:15–3:30  source map、bounded evidence、deterministic repair 三条主线
3:30–4:20  一个真实故障案例，例如 Miner window 或 HTML 0-chunk
4:20–5:00  canonical partial、已验证路径和仍不能声称的结论
```

建议只深挖三条主线：

### 主线一：为什么 source map 必须在模型前建立

模型只负责语义字段。parser 在本地维护 exact character span；line、source ID、
citation binding 和 evidence locator 也由本地代码从原文确定。`AtomicClaim` 不持有
offset，repair 会从可信 claim/source map 重新定位并换算 byte offset。这样避免模型
伪造位置，也让 terminal、SARIF 和 repair 共享可信坐标。

### 主线二：为什么 Agent 外面还要有 Controller

Controller 才拥有工具、预算、retry 和 side effects。Multi-Agent 如果没有确定性
控制面，会变成多个模型互相传自由文本，无法证明谁调用了网络、谁可以写文件，也
无法在单个 Agent 失败时安全继续。

### 主线三：如何安全地从“审计”走到“修改”

最终 verdict 不是直接交给另一个 Repairer Agent。确定性 planner 只从 claim 和
exact evidence 中提取唯一 scalar pair；随后检查类型、单位、位置、重叠和 SHA。
写回时先在内存验证 forward/reverse round trip，再同目录原子替换。这样 human
批准的是一个可定位、可恢复的最小变更。

## 12. Agent 工程岗位可重点讲

- typed Agent contracts 和 strict validation；
- plan validation 与 Controller fallback；
- per-claim failure isolation；
- bounded contexts、budgets 和一次 recovery；
- Scout snippet/evidence 分离；
- Judge/Challenger 职责与高风险门控；
- canonical Agent trace 和安全 telemetry；
- 为什么没有新增 Repairer Agent。

## 13. 后端工程岗位可重点讲

- canonical path、symlink、alias 和 output collision preflight；
- SSRF、redirect、DNS/IP 和 credential-bearing URL 防护；
- content hash、policy/version provenance，以及尚未接入当前 self-use 主链的 cache
  contract 边界；
- atomic file replace、stale detection、file-level isolation；
- canonical JSON、Markdown、terminal、SARIF 一致性；
- exit code 和 partial/fatal failure semantics；
- 隐私最小化与 payload-free diagnostics。

## 14. 推荐代码讲解顺序

1. `src/evidencetrace/cli.py`：产品入口和 `check`/`fix` 边界；
2. `self_use_inputs.py`：输入、路径和输出碰撞检查；
3. `markdown.py`：source-mapped Markdown；
4. `agents/miner.py`：bounded windows 与 exact claim；
5. `product.py`：Controller、计划校验和 Agent 编排；
6. `local_evidence.py`、`retrieval/`：本地与网页 evidence；
7. `repair.py`：scalar eligibility、stale 和 atomic write；
8. `self_use.py`：交互状态、batch manifest 和 per-target artifacts；
9. `sarif.py`：canonical audit 到 CI finding。

不要从测试数量或代码行数开始。先讲用户问题和最重要的不变量，再用代码证明。

## 15. 常见追问与短答

### 为什么不用一个大 prompt 一次完成？

单 prompt 会把 extraction、evidence、verdict 和修改权限混在一起，难以限制上下文、
定位失败和审计工具调用。拆分 Agent 后，每个 schema、预算和失败状态都可以独立
验证；Controller 防止职责拆分变成权限扩散。

### 为什么不用 LangGraph？

当前流程只有有限 initial/review 计划、固定预算和明确 side effects，直接 Python
Controller 能把权限、retry、fallback 和状态写成可审查代码。代价是编排代码更长；
如果未来真的出现复杂分支和持久化工作流，再以实际需求评估框架。

### 五个 Agent 是不同模型并行运行吗？

不是。五个 Agent 是职责和 typed-contract 边界，当前由 Controller 顺序、有界地
调度，也可以共享同一个 OpenAI-compatible client；不能描述成五个进程、分布式自治
或开放协作。

### 为什么不直接用向量数据库？

MVP 的 evidence volume 是单 citation/reference 内的小规模 chunks。FTS5 只筛选
Latin candidates，所有 candidates 由本地 BM25-style scorer 排序；无 FTS5 时扫描
全部 chunks，CJK 路径也扫描全部 chunks并加 n-gram。它不需要 embedding key且可
复现；真实规模和语义召回需求出现后，再用评测证据决定是否引入 dense retrieval。

### 怎么防止 hallucination？

不能证明完全没有 hallucination。项目做的是 exact claim、bounded evidence、
snippet 禁用、typed verdict、deterministic conflict、Challenger 和
`needs_human`。这降低并暴露风险，不等于真值证明。

### 模型输出坏 JSON 怎么办？

先区分 JSON envelope、strict Pydantic schema 和 local semantic validation。只有
schema error 可以进行一次 schema-only recovery；scope、evidence、budget 和 local
validation failure 不重试，也不做 JSON repair 或字段补全。

### 为什么 Challenger 不是每条 claim 都跑？

它只服务高风险 verdict。全量运行会增加成本和新的失败面；conditional one-shot
既提供独立复核，又保持每 claim 预算有界。

### 为什么 repair planner 不做成 Agent？

第一版只允许数字、日期和 SemVer 的最小替换。这个问题可以由确定性 token、type、
unit、位置和 diff 检查完整描述；再增加 Agent 只会扩大非确定性和写文件风险。

### 如何避免覆盖用户刚修改的文件？

开始时记录 target/reference SHA，apply 前重新读取和比较；target stale 时跳过该
文件，reference stale 时失效依赖候选。路径还经过 symlink/identity 检查。

### 一半文件写成功、一半失败怎么办？

产品采用 per-file isolation，不宣称 batch 全局事务。每个文件先完成内存
round-trip，再单独 atomic replace；失败文件保持原字节，manifest 明确记录 partial。

### Multi-Agent 比 Single Agent 更好吗？

目前没有证据支持质量优势。项目证明的是职责、权限、失败和 artifact 的工程边界；
历史 full-document smoke 也明确禁止推出 superiority 结论。

### 没有 API key 时还能运行什么？

Model-backed Agent 是可选路径。没有模型 credential 时，Controller 使用确定性的
Miner/Judge/计划与 Challenger fallback，仍能做 parser、retrieval、policy 和
artifacts；没有 Tavily key 时 discovery 明确返回 unavailable。不能暗示每次命令
都会执行五次 LLM 调用。

### `check` 既然只读，为什么还会生成文件？

“只读”限定的是用户输入：不修改 target、reference 或 Git index。为了可审计，
`check` 仍会原子写入 run artifacts；它不是零 filesystem write。

## 16. 演示建议

面试现场优先使用离线、确定性的路径：

```bash
uv run evidencetrace demo
uv run evidencetrace check --help
uv run evidencetrace fix --help
```

如果展示已有 artifacts，使用保存的 canonical run，不临时重跑历史 live/consumed
评测。已授权且已消费的临时副本 acceptance 不得重跑；任何再次真实 live
acceptance 都需要新的明确授权，不应为面试现场效果破坏既有证据边界。

## 17. 禁止夸大的表述

不要说：

- 自动修复任意事实；
- production-ready；
- 完整事实审计或高 recall；
- Multi-Agent 已证明优于 Single Agent；
- Tavily snippet 可以直接作为证据；
- Agent 可以自主写文件；
- Phase 3、Miner P0 或 latest partial 已解决；
- 已完成 PyPI/public Action 发布；
- 重新跑 consumed holdout 可以继续优化。

## 18. 面试前自检

在面试前确认自己能回答：

- 五个 Agent 分别做什么，Controller 为什么不是 Agent；
- source map 为什么由本地代码拥有；
- citation、reference、Tavily 的信任顺序；
- `needs_human` 在 `check` 和 `fix` 中有什么不同；
- strict schema 与 local validation 为什么不能混为一谈；
- Miner bounded windows 解决了什么，又没解决什么；
- atomic write、stale 和 reverse diff 如何保护用户文件；
- 为什么最新历史 run 是 partial，但 interactive fix 又可以 accepted；
- 哪些指标没有测量，哪些结论不能说。
