# EvidenceTrace 真实问题与工程复盘

本文不是“问题已经全部解决”的成功故事，而是一组可在面试中复核的工程案例。每个
案例都按以下顺序说明：

> 现象 → 根因 → 处理 → 为什么这样处理 → 如何验证 → 剩余边界

规范顺序是用户最新明确决定，其次是 Projects 根目录的
`evidencetrace_self_use_mvp_plan_2026-07-28.md` v2.0；当前实现以代码和 canonical
artifacts 为准。历史测试或评测数字只代表当时记录，不是本次文档工作的重新验收。

## 1. Inline code 在进入 Miner 前丢失

### 现象

第一轮真实 Miner 调用中，ADR 的版本号和 PR 文档的 `pull_request_target` 等关键
技术标识符没有形成 claim。历史 dogfood 记录显示，同一组三个案例的 17 次 dispatch
有 11 次失败。

### 根因

Markdown plain-text 预处理把 inline code 一并去掉。模型根本没有看到这些 token，
因此继续调 prompt、增加 token 或 retry 都无法恢复。

### 处理

- 非 URL inline code 保留在模型可见文本；
- inline-code URL 继续从模型可见文本移除，也不加入 citation 集合；普通 Markdown
  citation 仍由本地 parser 绑定；
- Miner failure code、attempt、schema recovery 和 finish reason 进入安全诊断。

### 为什么这样处理

这是确定性输入损坏，不是模型能力问题。应修 source-map/parser 边界，而不是用更多
模型调用掩盖。

### 验证与结果

历史同案例复核中，Miner failure 从 11 降到 9，ADR 的两类版本事实恢复。

### 剩余边界

PR 仍然是 0 claims，说明修复只解决一类输入丢失。状态因此保持
`Miner P0=improved_but_blocking`。

### 证据入口

- `src/evidencetrace/markdown.py`
- `docs/dogfood/round_1_2026-07-27.md`
- `docs/miner-architecture-decision.md`

## 2. 单段 Miner 调用的故障爆炸半径太大

### 现象

早期一个 paragraph 对应一个模型调用。PR 长段落在 schema recovery 后仍以
length/empty content 结束，唯一改动段落得到 0 claims；另一个案例中，一个
protected-token omission 会丢弃整段草稿。

### 根因

claim validity、paragraph coverage 和 provider failure 被绑在同一个大调用上。任何
一项失败都会删除同段其他有效结果。

### 处理

- 调用前规划 source-mapped bounded windows；
- 每个 window 独立 dispatch 和校验；
- exact monotonic mapping、protected occurrence 和 fragment validation 留在本地；
- window validity 与 paragraph coverage 分开；
- 不完整段落产生 ET2001/`needs_human`；
- 每文档最多 30 windows、60 次 Miner provider attempts，并继续受外层 provider
  budget 约束。

### 为什么这样处理

它从结构上缩小单次错误的影响，而不是通过重复调参赌一个更好的输出；失败窗口也不会
删除其他窗口已经成立的 exact claims。

### 验证与结果

历史 live recheck 中，PR 从 0 恢复到 4 claims，8 个 windows 中有 5 个完成；三份
案例共保留 14 个 exact claims。

### 剩余边界

ADR chosen option、PR trigger/write-access 和完整安全警告仍有缺失。该结果证明失败
隔离改善，不证明 Miner recall 已达标。

### 证据入口

- `src/evidencetrace/agents/miner.py`
- `docs/miner-architecture-decision.md`
- `docs/miner-coverage-failure-diagnosis.md`

## 3. Schema-valid 的空输出形成静默覆盖盲点

### 现象

包含 `pull_request_target` 的 window 可以返回合法的 `claims=[]`，旧逻辑却把它标为
complete；用户看不到核心技术事实可能被漏掉。

### 根因

protected coverage 主要观察数字、日期、版本、否定和比较词，没有把可信 inline-code
identifier 作为“这个窗口可能包含核心事实”的 coverage signal。

### 处理

parser 增加 identifier source-map kind；identifier window 的合法空输出转成
`miner_coverage_empty_identifier_window`，该 window 形成 `needs_human`；
paragraph 再根据其他 windows 汇总为 `partial` 或 `needs_human`。

### 为什么这样处理

确定性代码可以证明“窗口没有被覆盖”，却不能证明应该提取哪条 claim。将它标为
partial 比在本地凭规则生成事实更诚实。

### 验证与结果

历史记录为 165 个 focused tests、601 个 boundary-compliant tests 通过，coverage
policy 升级到 v2 并进入版本化 provenance。该 checkpoint 使用 0 次 live call，也
没有重新跑三份 dogfood，因此真实 extraction 是否改善仍未知。

### 剩余边界

这是 visibility fix，不是 extraction fix；它不代表 recall 提升，也不会替 Miner
生成 claim。

### 证据入口

- `src/evidencetrace/markdown.py`
- `src/evidencetrace/agents/miner.py`
- `src/evidencetrace/audit_models.py`
- `docs/miner-coverage-failure-diagnosis.md`

## 4. HTML 噪声规则把 citation 正文删成 0 chunks

### 现象

safe fetch 已成功，但 Python 官方冻结页面被 extractor 处理成 0 chunks，citation
evidence 无法交给 Judge。

### 根因

旧 noise regex 在属性值里做任意子串匹配，包含字符 `nav` 的普通属性也会被当成
navigation wrapper。

### 处理

改为 ASCII token-boundary noise matching，同时继续过滤真正的 nav/header/footer
结构。

### 为什么这样处理

问题在确定性 extraction 假阳性，不在 Agent。修正 token boundary 能保留现有安全
策略和 exact-span 证据要求，无需扩大模型权限。

### 验证与结果

冻结页面从 0 恢复到 32 chunks，三个 cited claims 各获得 5 个 bounded candidates。
canonical run `20260728T065626Z-40dc78f9` 中三个 citation claims 都进入 Judge，
safe fetch 总数为 6。

### 剩余边界

该 canonical run 仍是 `document_status=partial`，保留 Challenger、Judge 和 Scout
errors。准确结论只能是 `citation_handoff=repaired`，不能说整轮 check 已通过。

### 证据入口

- `src/evidencetrace/retrieval/extract.py`
- `tests/test_extraction_retrieval_checks.py`
- `.evidencetrace/runs/20260728T065626Z-40dc78f9/`

## 5. Judge 把“模型答错”误当作“系统完整性失败”

### 现象

schema-valid、in-scope 的 `not_checkable` 只因与本地 heuristic 不一致就抛异常，整轮
评测失败且普通模型质量错误无法被计分。

### 根因

local validator 越权决定语义 label，把 prediction disagreement 与 ID、scope、
evidence span 等 integrity failure 混为一谈。

### 处理

合法 relation 保留为 prediction；只有身份错配、source/span 不存在、`JudgeInput`
source-availability metadata 冲突和确定性 factual conflict 继续 fail closed。

### 为什么这样处理

评测必须区分：

- 模型给出一个合法但质量差的答案；
- 系统已经无法证明答案属于当前 claim/evidence。

前者应进入指标，后者才应中止。

### 验证与结果

历史 dev run 随后可完整执行并对语义错误计分；相关指标仍被标为 provisional。

### 剩余边界

这只修复 failure classification，不提升 Judge 本身的语义质量，也不把 dev 结果变成
正式 blind holdout。

### 证据入口

- `src/evidencetrace/agents/judge.py`
- `docs/implementation_status.md`

## 6. 让模型回传本地 ID 造成无意义 scope failure

### 现象

一次 Judge schema recovery 已成功，但模型回传的 `claim_id` 与请求不一致。系统正确
fail closed，却暴露出 contract 让模型负责了不必要的字段。

### 根因

旧 schema 要求模型 echo claim/source ID、locator、corroboration 和 contract
version。这些都是本地已经知道的事实，不需要语义推理。

### 处理

`LiveJudgeSemanticOutput` 只保留 relation、confidence、reason 和 evidence span；
本地从 trusted `JudgeInput` 组装 identity/provenance，并验证 exact span。

### 为什么这样处理

最小化模型所有权：模型只负责语义，本地代码负责身份和关联。这样减少 schema-valid
但对象错误的输出面。

### 验证与结果

历史三次 72-case author-constructed dev run 在该调整后全部完成；first-attempt
contract success 为 500/504，四次 schema failure 都由唯一 bounded recovery 恢复，
最终 operational failure 为 0。

### 剩余边界

这些是 pair-level provisional dev 证据，不运行完整 Claim Miner，也不证明
Multi-Agent 相对 Single Agent 的质量优势。

### 证据入口

- `src/evidencetrace/audit_models.py`
- `src/evidencetrace/agents/judge.py`
- `docs/implementation_status.md`
- `docs/phase3g-architecture-decision.md`

## 7. Provider strict schema 失败既不稳定，也难以诊断

### 现象

历史 full-document smoke 的一个 logical call 做了两次 provider attempts，最终仍以
`model_schema_invalid` 失败。两次共报告 4,096 output tokens，在每次 2,048 ceiling
下饱和，但不能证明 truncation。

### 根因

已确认的问题是 full-document contract 缺少可诊断字段，且旧 2,048 上限不能排除
budget 影响；历史失败的具体根因仍是 `unknown_schema_failure`。

### 处理

- full-document contract 固定 `max_tokens=8192`；
- model reason 限 240 字符；
- 记录 payload-free 的 stage、field path、usage、content length 和 truncation
  signal；
- 仅对 `ModelSchemaError` 做一次固定 schema reminder recovery；
- 不做 JSON repair，不回灌失败响应，不补造语义字段。

### 为什么这样处理

8,192 是新的 full-document contract 工程上限，不是已证明的根因修复。先提高
contract 可诊断性，再决定真正原因；无限 retry 或继续加 token 会消耗真实调用，
却不能证明为什么偶然成功。

### 验证与结果

后续独立 bounded smoke 在同一个 frozen dev 文档 `fdv1_doc_001` 上运行：Single
Agent 第一次能定位到 `claims/0` 的 value error，`finish_reason=stop` 且明确不是
truncation，唯一 recovery 成功；Multi-Agent 的 4 次 Miner 与 5 次 Judge 都首次
通过。合计使用 11/20 attempts、18,597/200,000 tokens。

### 剩余边界

这只是一篇文档的 operational smoke，不是 full-dev quality，更不是 Multi-Agent
优势证明。本次文档整理没有新增 provider 调用。

### 证据入口

- `src/evidencetrace/model_client.py`
- `src/evidencetrace/eval/full_document.py`
- `docs/phase4b-provider-contract.md`
- `docs/phase4b-live-dev.md`

## 8. Citation discovery 与 evidence 资格不能混在一起

### 现象

Tavily snippet 往往已经“像答案”，若直接交给 Judge 或 repair，就无法证明页面真实
内容、上下文和 locator。

### 根因

搜索 ranking signal 与 evidence provenance 被混为同一种数据。

### 处理

Scout 只输出受预算限制的 query/URL candidates。搜索服务可以返回 snippet，但当前
Controller 路径只消费候选 URL；snippet 不进入 evidence payload。每个 URL 仍需
safe fetch、extract、rank 和 exact span。Tavily-only evidence 在交互式 `fix` 中
还要独立 human trust，apply 另行确认。

### 为什么这样处理

把“去哪找”与“找到了什么”分开，避免第三方摘要被提升成事实证据，也保留人对联网
来源的最终信任权。

### 验证与结果

真实 Tavily path 已达到 `tavily_live_path=verified`，但该字段只验证受控 discovery
链路，不等于相关 claim 或整个文档都通过。

### 剩余边界

无 key、无合格 URL、safe-fetch 拒绝或动态网页抽取失败都会保持 unavailable/partial。

### 证据入口

- `src/evidencetrace/agents/scout.py`
- `src/evidencetrace/product.py`
- `src/evidencetrace/self_use.py`

## 9. 从“判断能改”到“文件敢改”

### 现象

`check` 能发现 contradiction，不代表 Agent 可以直接重写文件。自由改写可能改变语义、
单位或范围；apply 时还可能覆盖用户刚做的编辑，或者进程中断留下半写文件。

### 根因

语义判断、变更规划、用户授权和文件系统写入原本没有被拆成独立安全门。

### 处理

- 不新增 Repairer Agent；
- deterministic planner 只支持 integer/decimal/percentage、四种 allowlisted date
  和 SemVer；
- 要求 exact evidence、final contradiction、同类型/单位，以及 Challenger typed
  `uphold`；若为 `revise`，修订后仍须由同一 exact evidence 支持同一 replacement；
  `abstain`、error 或 incomplete 都不可修；
- TTY 中分别确认 evidence/option 和 apply；
- 批准集合在真正写文件前还有最终 write confirmation；
- apply 前复核路径身份、SHA、位置、overlap 和禁止区域；
- 先验证 forward/reverse round trip 并保存 reverse diff；
- 同目录临时文件、`fsync`、`os.replace`；
- 每个 target 独立 outcome。

### 为什么这样处理

Agent 适合做语义比较，不适合拥有无边界的文件写权限。确定性最小 diff 才能把“证据
支持的结论”转成“可预览、可恢复的字节修改”。

### 验证与结果

已明确授权且只运行一次的真实 PTY 临时副本 acceptance 已完成，状态是
`interactive_fix=accepted`。它是两个只读 canonical runs 之后的独立 checkpoint，
不修改真实项目或用户目标文件；任何再次真实 live acceptance 都需要新的明确授权。

### 剩余边界

两个 `.evidencetrace` canonical runs 都是更早的 read-only `check`，仍为 partial，
并不承载 acceptance 证据。`candidate_patch=unverified` 指历史 check 的候选 patch，
不能与之后的 temp-copy acceptance 合并；`overall_new_self_use_goal=incomplete`
继续保留。accepted 不代表任意事实可自动修复、production-ready 或历史 check 状态
被覆盖。本次文档工作没有继续重跑。

### 证据入口

- `src/evidencetrace/repair.py`
- `src/evidencetrace/self_use.py`
- `src/evidencetrace/cli.py`
- `docs/PROJECT_HANDOFF.md`
- Projects 根 `evidencetrace_self_use_mvp_plan_2026-07-28.md` §12、§14

## 10. 评测投入没有自动形成用户闭环

### 现象

项目积累了 pair benchmark、holdout 和 full-document smoke，但 pair 数据不运行
Claim Miner；已消费 holdout 上继续调整也不能证明泛化。与此同时，多文档 check、
local reference、统一 artifacts 和安全 fix 才是自用所需闭环。

### 根因

“有指标”被过度等同为“产品可用”，而 extraction、证据、交互和写回没有在同一真实
工作流中验收。

### 处理

DD-0050 冻结扩评：不新建 v5、不重复消费 frozen holdout；Phase 3 保持
`completed_with_known_limitations`、`phase4_eligible=false`，工作转向 self-use
vertical loop。

### 为什么这样处理

评测应回答明确问题，不能替代端到端产品事实。已消费数据上继续取得更好数字，反而会
削弱结论可信度。

### 验证与结果

多目标输入、local references、canonical per-target artifacts、受控 discovery 和
interactive scalar repair 已形成工程路径；当前状态仍明确分层记录。

### 剩余边界

没有正式 blind-holdout F1，没有完整 extraction quality gate，也没有证明
Multi-Agent 质量优于 Single Agent。

### 证据入口

- `docs/design-decisions.md`
- `docs/implementation_status.md`
- `docs/full-document-benchmark-design.md`
- `docs/known-limitations.md`

## 11. 面试时如何讲这些问题

优先选择两个能互补的案例：

1. Miner bounded windows：展示 Agent contract、failure isolation 和 coverage；
2. deterministic fix：展示后端路径安全、human authorization 和 atomic write。

如果面试官继续追问：

- 模型接口：讲 provider strict schema 和 semantic/local ownership；
- Retrieval：讲 snippet 不算 evidence、safe fetch 和 HTML 0-chunk bug；
- 评测方法：讲 Judge quality/integrity 分类与冻结已消费 holdout；
- 诚实边界：主动说明 latest canonical run 仍是 partial、Miner P0 未通过。

一个合格的复盘回答应同时包含根因、为什么没选更简单的“多 retry/多 token”方案、
验证证据以及剩余限制，不能只说“后来修好了”。
