# EvidenceTrace 组件深挖

本文按统一模板说明当前产品主链的每个部分：

> 作用 → 使用的手段 → 为什么这样选 → 问题与边界 → 代码入口

它是面试说明材料，不是新的规范计划。规范以 Projects 根 v2.0 计划为准，实现事实
以当前代码和 canonical artifacts 为准。

## 1. CLI 与模式门

### 作用

把只读审计和有副作用的修复做成两个明确入口：

- `check`：一个或多个目标，只读；
- `fix`：同一审计链路之后，进入交互式 repair；
- `demo`：离线、byte-stable 演示；
- `eval`：历史评测入口，不属于当前自用闭环。

普通单文档 Markdown `check` 还保留 legacy 路径和 `--changed-from` changed-paragraph
selection。多目标、TXT、reference、discover/output-dir 与所有 `fix` 进入 self-use
batch；Git diff 只用于只读选择，绝不是可写 target。

### 使用的手段

- Typer 声明参数和命令；
- `fix` 启动前检查 interactive TTY；
- `--discover` 是联网授权，不是默认行为；
- 不提供 `--yes`。

### 为什么这样选

写权限不能只靠 prompt 约束。把 `check` 和 `fix` 分开，使“谁可能写文件”在 CLI
入口就可验证；TTY 和无 `--yes` 阻止后台、CI 或脚本静默批准修复。

### 问题与边界

- `fix` 不能无人值守运行；
- non-TTY 只能安全拒绝或保持只读；
- CLI 模式门只是第一层，后续仍需 evidence、stale 和 atomic-write 检查。

### 代码入口

- `src/evidencetrace/cli.py`

## 2. 输入 preflight 与路径安全

### 作用

在模型或网络调用前冻结输入边界，拒绝可能造成路径混淆、覆盖或隐私泄露的请求。

### 使用的手段

- 最多 20 targets、50 references、合计 5 MiB；
- 多目标、全部 `fix` 和 references 只接受显式普通 `.md`、`.markdown`、`.txt`；
- 普通单目标 Markdown `check` 保留 legacy 兼容入口；
- 拒绝目录、symlink、非普通文件和非 UTF-8；
- 用 canonical path、device/inode 检测重复与 alias；
- target、reference、显式/派生 output 不得碰撞；
- 为 artifacts 生成 safe ID，记录内容 SHA，不暴露用户主目录绝对路径；
- 使用 `lstat`、`resolve`、`stat` 和 no-follow 检查。

### 为什么这样选

如果输入边界在 Agent 调用后才验证，既浪费调用预算，也可能让私有路径、重复文件或
错误输出目标进入 prompt/artifact。先冻结文件身份，还为 apply 前的 stale 检查提供
基线。

### 问题与边界

- 不递归目录；
- shell glob 必须由调用者先展开；
- symlink 即使指向安全普通文件也拒绝；
- 路径身份仍会在真正写回前再次验证，preflight 不能单独消除 TOCTOU。

### 代码入口

- `src/evidencetrace/self_use_inputs.py`

## 3. Markdown/TXT parser 与 source map

### 作用

把模型可见文本中的每个可审计片段映射回原始文件 character offset 和行号；repair
阶段再把可信字符位置换算为 byte offset。

### 使用的手段

- Markdown 使用 markdown-it-py 取得 block/inline token；
- 本地 scanner 处理 citation、footnote、inline code、URL 和 suppression；
- 模型可见文本与 trusted source map 在同一 deterministic pass 中生成；
- TXT 按空行分段并维护行映射；
- 裸 URL 既可成为 citation candidate，也始终属于 protected span。

### 为什么这样选

模型擅长语义抽取，但不应拥有 locator。offset、line 和 source binding 如果由模型
返回，会出现行号幻觉、ID echo 错误和无法安全修复的问题。本地 source map 让 Miner、
SARIF 和 repair 使用同一坐标系。

### 问题与边界

- 不是完整浏览器或所有 Markdown extension 的实现；
- 早期 plain-text 预处理曾丢失 inline-code 事实；
- parser bug 会在整个下游放大，因此 parser 输出必须有独立回归测试。

### 代码入口

- `src/evidencetrace/markdown.py`
- `src/evidencetrace/local_evidence.py`

## 4. Deterministic Controller

### 作用

Controller 是产品的权限和状态所有者，负责：

- Agent 调度与 typed plan 校验；
- 工具、网络、预算和 retry；
- claim/window/file 级失败隔离；
- evidence 路由与 human gate；
- policy、exit code 和 canonical artifacts；
- repair eligibility、交互和写回。

### 使用的手段

- 明确状态模型而不是 Agent 自由文本接力；
- initial/review 两阶段计划；
- allowlisted action 和 deterministic fallback；
- per-document pipeline budget、per-claim/per-window limits 与 file-level
  isolation；
- partial/fatal 状态与跨字段 validation。

### 为什么这样选

Multi-Agent 解决职责隔离，但不能自动解决权限隔离。若每个 Agent 都能联网、读文件
或决定写回，就无法证明调用边界，也无法在某个 Agent 失败后安全继续。Controller
让 Agent 只负责受限语义判断。

### 问题与边界

- 流程比自由 Agent 更保守，`partial` 会更多；
- 状态模型和 artifact contract 更复杂；
- Controller 不是 Agent，不能为了数量包装成第六个 Agent。

### 代码入口

- `src/evidencetrace/product.py`
- `src/evidencetrace/self_use.py`

## 5. Miner

### 作用

从目标文本提取 atomic、source-located、逐字符一致的 claims；不读取 evidence，也不
判断真假。

### 使用的手段

- paragraph 先被 deterministic planner 切成 bounded windows；
- 每个 window 独立 dispatch；
- 模型只返回 claim text/type/checkability；
- source map 验证 exact character span；claim ID、line 和 citation 由本地组装；
- `AtomicClaim` 不持有 offset，repair 后续会重新定位并计算 char/byte offsets；
- exact substring、单调 non-overlap、protected occurrence 和 fragment 检查；
- window validity 与 paragraph coverage 分开；
- 不完整 paragraph 通过 ET2001/canonical outcome 可见。

### 为什么这样选

早期一整个长段落一次调用，一个截断、空结果或 protected-token omission 会吞掉整段
有效 claims。Bounded windows 缩小输出复杂度和故障爆炸半径，同时不放宽 exact
claim 边界。

### 问题与边界

- 默认最多 30 个 windows，每 window 最多两个 provider attempts；
- abbreviation boundary、oversized span、protected coverage 仍可能产生 partial；
- coverage-only identifier signal 只暴露空窗口，不会替模型创造 claim；
- Miner P0 仍是 `improved_but_blocking`。

### 代码入口

- `src/evidencetrace/agents/miner.py`
- `ProductAuditPipeline._mine` in `src/evidencetrace/product.py`

## 6. Coordinator

### 作用

产生 initial 和 review 两个 typed execution plan，在有限 action allowlist 中建议
claim 的下一步。

- initial：`verify_citation`、`discover`、`skip_not_checkable` 或
  `request_human`；
- review：`accept`、`challenge`、`counter_search` 或 `request_human`。

Local-reference 挂载、evidence routing 和 Judge 调度属于确定性 Controller，不是
Coordinator 决策。

### 使用的手段

- strict Pydantic plan；
- action allowlist；
- initial/review 各最多一次；
- Controller 检查 claim ID、阶段、action 与权限；ExecutionPlan 不拥有 source ID；
- Agent 不可用或 plan 越权时使用 deterministic fallback。

### 为什么这样选

Coordinator 展示真正的 Agent 编排判断，但不拥有工具。这样既能根据审计状态调整
计划，又不形成开放循环或“模型说调用什么就调用什么”的权限漏洞。

### 问题与边界

- 最多两次，不是自治规划循环；
- plan failure 会降低为保守 fallback；
- Coordinator 不能自行获取网页、读取文件或写回。

### 代码入口

- `src/evidencetrace/agents/coordinator.py`
- `ProductAuditPipeline._plan`

## 7. Citation 与网页 safe fetch

### 作用

把直接 citation 变成可定位、可验证的网页 evidence，同时限制 SSRF、secret 泄露和
非文本内容风险。

### 使用的手段

- URL scheme、userinfo、credential query 参数检查；
- DNS 解析和 private/loopback/link-local/metadata IP 拒绝；
- redirect 和重试时重新验证并 pin 目标；
- timeout、MIME、响应大小和文本编码限制；
- stdlib `HTMLParser` 去除 script/style/nav 等噪声；
- 输出 bounded exact chunks，而不是整页或 snippet。

### 为什么这样选

URL 是 Agent 工具面，不是普通字符串。先验证再请求比事后日志脱敏更安全；exact
fetched span 才有证据资格，搜索摘要不能替代。

### 问题与边界

- 不支持 signed/authenticated URL；
- 不支持登录页、动态浏览器渲染或 JavaScript；
- stdlib extractor 依赖少、测试稳定，但不是浏览器级正文抽取；
- 真实出现过 `nav` 子串误判导致 0 chunks 的 bug。

### 代码入口

- `src/evidencetrace/retrieval/fetch.py`
- `src/evidencetrace/retrieval/extract.py`

## 8. Local reference

### 作用

把用户显式提供的 Markdown/TXT 文件作为共享只读 evidence library。

### 使用的手段

- 复用 Markdown/TXT source map；
- 建立 bounded chunks 和 lexical index；
- evidence 保留 safe source ID、hash、line 和 exact span；
- artifacts 不复制完整私有文件或绝对路径；
- apply 前重新检查 reference SHA。

### 为什么这样选

项目自用场景中，规范、ADR 和 README 经常已经在本地。优先本地 evidence 可以减少
联网成本，也比让 Scout 搜索公开网页更贴近仓库事实。MVP 复用小规模 lexical
retrieval，避免引入向量服务。

### 问题与边界

- references 必须显式提供，不扫描目录；
- lexical retrieval 对同义改写有限；
- stale reference 会使依赖候选失效，不能用 human yes 绕过。

### 代码入口

- `src/evidencetrace/local_evidence.py`

## 9. Lexical retrieval

### 作用

从 citation/local-reference chunks 中选取和 claim 最相关的 bounded evidence。

### 使用的手段

- SQLite FTS5 只缩小 Latin/标识符候选；
- 所有候选始终由本地 BM25-style scorer 排序；
- SQLite 没有 FTS5 时扫描全部 chunks；
- Latin token 很少时使用 NFKC CJK 2/3-gram；
- ISO-ish 日期、百分比和 SemVer-ish 版本使用 exact special feature；普通整数/小数
  走常规 token score；
- 可带有限相邻 chunk，但总 context 有硬上限。

### 为什么这样选

MVP 中每个 source 的 chunk 数小，lexical retrieval 可离线、可解释，不需要
embedding credentials 或向量数据库。对 scalar fact，identifier、数字和版本的
词面信号尤其重要。

### 问题与边界

- 不是 dense semantic retrieval；
- CJK n-gram 不是成熟中文分词；
- 同义改写和长距离语义关系召回有限；
- 增加 top-k 不能解决“候选生成阶段完全 miss”的问题。

### 代码入口

- `src/evidencetrace/retrieval/rank.py`

## 10. Scout 与受控 discovery

### 作用

只为仍 unresolved 且用户传入 `--discover` 的 claim 规划搜索，并发现候选 URL。

### 使用的手段

- 每 claim 最多 2 query、5 URL、3 safe fetch；
- query 必须来自 claim 词项或 allowlist；
- SearchResponse 可携带 Tavily snippet，但当前 discovery 路径只消费 URL；
- URL 必须经过同一 safe fetch 和 exact extraction；
- 没有 key 或搜索失败时形成 `discovery_unavailable`/partial。

### 为什么这样选

联网 discovery 能补 citation/reference 缺口，但也是成本、隐私和提示漂移来源。
显式授权和硬预算比让 Agent 自由搜索更可审计。

### 问题与边界

- 召回受硬预算限制；
- snippet 不进入 Judge、repair 或 evidence payload；
- search/fetch failure 不得终止其他 claims；
- Tavily-only evidence 在 `check` 中仍需要 human。

### 代码入口

- `src/evidencetrace/agents/scout.py`
- `ProductAuditPipeline._discover`

## 11. Evidence router 与 human trust

### 作用

在 Judge 前确定哪一组 evidence 可以用于当前 claim，并暴露来源冲突。

### 使用的手段

- citation/reference 一致且无冲突时直接选择；
- 对候选 scalar signature 做确定性分组；
- Tavily-only 或来源冲突：
  - `check` → `needs_human`；
  - `fix` → human 只能信任 exact span，或选择已有 evidence-backed option；
- 不允许 human 自由输入新的 replacement。

### 为什么这样选

Human 可以决定“是否信任这个来源”或“冲突来源选哪个”，但不能替代 Judge
verdict，也不能创造 evidence。这样 human gate 不会退化成任意绕过按钮。

### 问题与边界

- 冲突 options 超过有界上限时保持 `needs_human`；无冲突候选溢出可截断后
  deterministic 择优，多 options 本身不等于跳过；
- Human trust 只解决来源选择，不能解决 Agent error、无 span、stale 或禁止位置；
- `check` 对 target/reference/Git index 保持只读并且 non-interactive；仍会写审计
  artifacts。

### 代码入口

- `ProductAuditPipeline._route_evidence`
- `InteractiveSession` in `src/evidencetrace/self_use.py`

## 12. Judge

### 作用

一次只比较一个 claim、一个 selected source 和 bounded evidence，输出 relation、
confidence、reason 和 evidence span。

### 使用的手段

- 模型只拥有语义字段；
- claim/source ID、line、locator 和 Judge semantic/ownership contract version
  由本地代码组装；
- strict Pydantic；
- evidence substring/scope 检查；
- 数字、日期、版本、entity、negation 等 deterministic guards；
- schema-valid 但语义质量不佳的 relation 作为 prediction 保留；伪造位置和本地确定
  冲突才 fail closed。

### 为什么这样选

早期让模型 echo 本地 ID/locator，出现过“语义输出正确但 ID 不匹配”的无意义 scope
failure。把 provenance 留给本地，模型只做真正需要语义判断的部分，缩小 provider
contract。

### 问题与边界

- Judge 不是 truth oracle；
- deterministic guard 可能保守地增加 partial；
- schema-valid relation disagreement 是质量问题，不应自动伪装成系统完整性错误。

### 代码入口

- `src/evidencetrace/agents/judge.py`
- `ProductAuditPipeline._judge`

## 13. Challenger

### 作用

对 contradiction、低置信、结构化事实和比较等高风险结论做一次独立复核。

### 使用的手段

- deterministic high-risk predicate；
- 每 claim 最多一次；
- typed `uphold`、`revise`、`abstain`；
- revised outcome 仍须使用已有 bounded evidence；
- counter-search 不能由 Challenger 自行触发，仍需 Controller/Scout 授权。

### 为什么这样选

不是所有 claim 都值得第二次模型调用。Conditional one-shot 在成本有界的情况下，为
高风险 repair 增加独立异议门。

### 问题与边界

- 它增加一个新的 schema/transport failure 点；
- `abstain` 或 error 不能 repair；
- Challenger 成功也不能绕过 scalar、location、stale 或 human apply gate。

### 代码入口

- `src/evidencetrace/agents/challenger.py`
- `ProductAuditPipeline._challenge`

## 14. OpenAI-compatible 模型契约

### 作用

统一 model-backed Agent 的 JSON schema 调用、预算、错误分类和安全 telemetry。

### 使用的手段

- strict Pydantic、required fields、`extra=forbid`；
- 请求 `temperature=0.0` 和显式 max tokens；provider 实际 effective temperature
  仍可能未知；
- JSON envelope 与 Pydantic schema 分阶段检查；
- 只有 `ModelSchemaError` 可以进行一次 schema-only recovery；
- 不修 JSON、不补字段、不纠正 enum、不做 semantic normalization；
- diagnostics 只保留 allowlisted stage/path/code、长度和 token 数，不保留响应正文。

### 为什么这样选

OpenAI-compatible provider 的 JSON mode 不等于 schema guarantee。一次受限 recovery
可以处理偶发结构错误，但开放 retry/repair 会掩盖 provider 不稳定和成本。

### 问题与边界

- transport、scope、evidence、budget 和 local validation failure 不重试；
- 一次 recovery 失败后按所在 Agent/路径聚合为 partial/fatal；
- safe diagnostics 能定位 contract stage，但不能重构缺失的 provider 内容。

### 代码入口

- `src/evidencetrace/model_client.py`

## 15. Deterministic scalar repair

### 作用

把最终 contradiction 转成最小、可验证的 scalar replacement，而不是自由文本改写。

### 使用的手段

仅支持：

- 整数、小数、百分比；
- 四种 allowlisted date surface：`YYYY-MM-DD`、`YYYY/MM/DD`、
  `Month D, YYYY`、`D Month YYYY`；
- SemVer。

候选必须同时满足：

- final relation 是 completed `contradicted`；
- Challenger typed `uphold` contradiction，或 `revise` 后仍由同一 exact evidence
  支持同一 replacement；`abstain`、error、incomplete 均不通过；
- replacement 来自 selected exact evidence；
- claim 与 evidence 各有唯一可映射 scalar；
- 类型、日期 family 和单位兼容；
- target span 允许且不 overlap；
- target/reference SHA 有效。

URL、fenced code、citation target、front matter 和 HTML attribute 等位置受保护。

### 为什么这样选

第一版 repair 是可由确定性规则完整描述的问题。新增 Repairer Agent 会引入自由
改写、位置幻觉和无法证明的 replacement，因此被明确拒绝。

### 问题与边界

- 不换单位；
- 不插 citation；
- 不修一般自然语言；
- evidence source conflict 可让 human 从 bounded existing options 中选择；
- claim/evidence 自身的 scalar ambiguity 直接保持 `needs_human`，不能靠交互自由
  选择 scalar。

### 代码入口

- `src/evidencetrace/repair.py`

## 16. 交互与原子写回

### 作用

让用户逐条决定是否应用候选，并保证写回可恢复、可审计且不覆盖 stale 文件。

### 使用的手段

- `yes/no/quit`；
- `quit` 后对之前批准项再做一次总确认；
- apply 前重算 target/reference SHA；
- 同一文件 edits 先在内存排序并检查 non-overlap；
- forward/reverse byte round trip；
- 写源文件前先持久化 reverse diff；
- 同目录 no-follow 临时文件、`fsync`、`os.replace`；
- 每文件隔离，失败文件保持原字节。

### 为什么这样选

“用户点了 yes”并不能解决 stale、symlink、partial write 或不可恢复问题。Human
authorization 与文件系统安全是两个独立门。

### 问题与边界

- batch 不是跨文件全局事务；
- 只能 interactive TTY；
- 文件元数据和平台原子语义仍受操作系统约束；
- 任何路径身份变化都应 fail closed。

### 代码入口

- `src/evidencetrace/self_use.py`
- `apply_repair_plan` in `src/evidencetrace/repair.py`

## 17. Canonical artifacts、Markdown 与 SARIF

### 作用

让终端、JSON、Markdown、SARIF 和 diff 表达同一个 canonical outcome。

### 使用的手段

- 每 target 的 `audit.json` 是 canonical truth；
- `batch-manifest.json` 只索引 targets/references/status；
- `audit.md`、terminal 和 SARIF 从同一已验证 typed artifact 派生；
- 三类 diff 来自 deterministic RepairPlan/ApplyResult，并与 canonical repair/apply
  decisions 保持一致；
- strict cross-field validators；
- stable SARIF rule、location、fingerprint 和排序；
- atomic artifact writes；
- safe IDs 和最小隐私字段。

### 为什么这样选

如果每个 renderer 使用不同的运行状态，就可能出现 terminal 成功、JSON partial、
SARIF 又是另一组 finding 的情况。当前各视图由同一个已验证 typed canonical state
派生，并把 `audit.json` 持久化为事实记录，使 CI 和人工审查共享一个状态源；不能
泛化成 batch 主链总会先写后重读 JSON。

### 问题与边界

- artifact model 较复杂，写出失败也需要 fail closed；
- legacy 单目标和当前 batch/fix layout 不同；
- canonical artifact 是系统运行事实，不是真值或质量证明。

### 代码入口

- `src/evidencetrace/self_use.py`
- `src/evidencetrace/artifacts.py`
- `src/evidencetrace/render.py`
- `src/evidencetrace/sarif.py`

## 18. Cache、config 与评测模块的准确边界

### 作用

保存历史评测基础设施、版本化 contract 和可复用辅助模块，同时明确哪些没有进入
当前 self-use/product 主链。

### 使用的手段

- `cache.py` 定义带 policy/version dimensions 的 SQLite cache key 与 `CacheStore`；
- `config.py` 定义 TOML loader；
- `eval/`、`eval_sets/` 和 `eval_runs/` 保存评测逻辑、输入与 provenance。

### 为什么这样选

这些历史工程帮助形成 strict schema、fail-closed、ownership 和 provenance 决策，
也保留可复核记录；但“代码文件存在”不能替代从 CLI 到主链的调用事实。

### 问题与边界

当前 self-use/product `check`/`fix` 主链根本没有调用 `CacheStore`，既不读也不写
该 cache；CLI 也没有接入 TOML loader。面试时可以说“实现过版本化 cache/config
辅助模块”，不能说当前路径会命中 cache、执行 TTL 或读取自定义 TOML。历史
dev/provisional 指标也不能当成正式产品质量。

### 代码入口

- `src/evidencetrace/cache.py`
- `src/evidencetrace/config.py`
- `src/evidencetrace/eval/`
