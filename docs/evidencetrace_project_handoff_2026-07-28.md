# EvidenceTrace 项目介绍与新窗口交接

版本：1.0  
日期：2026-07-28  
用途：让一个没有聊天历史的新窗口独立理解项目并安全继续工作  
建议 Mac 路径：`~/Projects/evidencetrace/docs/PROJECT_HANDOFF.md`

## 0. 新窗口先读这里

EvidenceTrace 的最新代码只在用户 Mac：

```text
~/Projects/evidencetrace
```

当前对话执行环境中的服务器 repo 副本是旧版本，不能据此继续开发。本文件是传输
版本；复制到 Mac 后建议保存为 `docs/PROJECT_HANDOFF.md`。

新窗口必须按以下顺序开始：

1. 完整阅读本 handoff；
2. 完整阅读仓库根目录的
   `evidencetrace_self_use_mvp_plan_2026-07-28.md`；
3. 在 Mac 仓库执行只读的 `git status`、`git log -1`、`git diff --stat`、
   staged diff 和 remote 检查；
4. 定位最新 canonical run artifacts；
5. 先复述项目目标、当前状态、禁止事项和唯一下一任务；
6. 先给阶段 0 只读诊断，再修改代码。

不要从旧 handoff 中继续“实现 Coordinator/Scout/Challenger”，它们已经存在。
不要从旧发布计划继续 PyPI、GitHub Action、10 案例或高-star包装。

## 1. 权威顺序

规范性权威：

1. 用户最新明确决定；
2. `evidencetrace_self_use_mvp_plan_2026-07-28.md` v2.0。

实现与实测事实的证据顺序：

1. Mac 最新代码与 canonical artifacts；
2. 本 handoff；
3. 历史 `multi_agent_researchops_project_plan.md`、Miner ADR、Round 1、coverage
   diagnosis；
4. 2026-07-25 `evidencetrace_handoff.md`，仅作历史。

本 handoff 是描述性快照；它过时时不能覆盖 Mac 上可复核的代码/artifacts，服务器
旧副本也永远不能用于判断当前实现。历史评测结论、安全不变量和失败证据仍有效，
除非新计划使用明确标记调整。当前调整标记是：

```text
[2026-07-28 Interactive Evidence-Grounded Fix Scope Adjustment]
```

## 2. 一句话项目定义

EvidenceTrace 是一个证据驱动、受 Deterministic Controller 约束的 Multi-Agent
Markdown/TXT 事实审查与交互式修复 CLI：

```text
多个目标文档
→ Miner exact claims
→ Coordinator typed plan
→ citation + 本地 references
→ 必要时 Scout/Tavily
→ safe fetch + bounded evidence
→ deterministic conflict / Tavily trust gate
→ 必要时 human evidence resolution
→ Judge
→ Coordinator review
→ conditional Challenger
→ Policy + canonical audit
→ deterministic scalar repair candidate
→ separate apply confirmation
→ atomic write + reversible diff
```

`check` 永远只读。`fix` 只在交互 TTY 中、经过证据门控和逐条确认后修改原文件。

## 3. 用户真正想要什么

用户目标岗位：

- Agent 工程；
- 后端工程。

项目优先证明：

- 真实 Multi-Agent 职责隔离；
- typed contracts；
- deterministic Controller；
- tool/permission/budget 边界；
- failure isolation；
- citation/local/Web evidence；
- 可审计和可恢复的文件修改；
- 诚实 partial 与 known limitations。

最低产品要求：本人可以实际使用。当前不要求演示、公开 GitHub、高 star 或 PyPI。
当前时间边界为 1–2 个工作日。

## 4. 当前与未来的 CLI

当前已有单目标 `check` 产品路径和 evidence-gated candidate diff。最新规范要求：

```bash
uv run evidencetrace check TARGET1.md TARGET2.txt \
  --reference REF1.md --reference REF2.txt --discover

uv run evidencetrace fix TARGET1.md TARGET2.txt \
  --reference REF1.md --reference REF2.txt --discover
```

计划中的最终行为：

- `check`：多目标、只读；
- `fix`：同一审计路径后逐条确认并写回；
- targets/references 都支持多个显式 MD/TXT；
- 现有单目标 `check` 对 Markdown/README/ADR/Git diff 继续只读兼容；新增多目标和
  全部 `fix` 只处理普通 `.md/.markdown/.txt`，不写回 Git diff；
- shell 可展开 glob，但 CLI 不递归目录；
- 非 TTY `fix` 不写文件；
- 不提供 `--yes`。

单目标继续兼容 `--sarif FILE`、`--suggest-patch FILE`。多目标使用默认 run
root/per-target artifacts，或通过 `--output-dir DIR` 指定 run root；多目标若混用
单文件输出参数，必须在任何模型/网络调用前拒绝。所有 output paths 还必须
canonical-resolve，且不得与 target/reference 或其他 output 碰撞。TXT target 是
UTF-8 plain text，
按空行分段并保留行定位；裸 `http(s)` URL 可作为受控 citation 候选且其范围禁止
repair。

多目标、本地 reference 和交互式写回是**待实现**，不能写成当前已有能力。

## 5. Agent 与 Controller

### 5.1 Miner

- 从模型可见文本提取连续、逐字符一致的 atomic claim；
- 保留 citation、位置、限定词、数字、日期、版本、单位、否定与比较；
- 不看 evidence，不判断真假，不修改文本。

### 5.2 Coordinator

- 每文档最多 initial plan 和 review plan 两次；
- 只输出预定义 typed actions；
- 无网络、文件、shell 工具；
- 不能创建 Agent、prompt、action、预算或循环；
- 越权计划由 Controller 拒绝并执行 deterministic fallback。

### 5.3 Scout

- 只处理被授权的 unresolved claim；
- 每 claim 最多 2 query、5 candidate URL、3 safe fetch；
- Tavily snippet 只能选择候选 URL，不能作为 evidence；
- URL 必须通过现有 safe-fetch 和 evidence pipeline。

### 5.4 Judge

- 每次只看一个 claim 和 bounded evidence；
- 没有搜索工具；
- 实质 verdict 必须引用真实 evidence span；
- 无可用 bounded span 时不得猜测；Judge 返回现有 typed non-complete/abstain，
  Controller 再映射为 `needs_human` 或 operational outcome。`not_in_source` 只按
  既有 typed contract 的特定语义使用，不能充当通用“缺证据”状态。

### 5.5 Challenger

- 只复核矛盾、低置信度、数字、日期、版本和比较等高风险 claim；
- 每 claim 最多一次；
- 只能 uphold/revise/abstain；
- counter-search 必须由 Coordinator 授权 Scout。

### 5.6 Deterministic Controller

Controller 不是 Agent。它拥有：

- 权限、预算、并发、状态、timeout、retry；
- tools、safe fetch、local retrieval；
- artifact、cache、policy、exit code；
- Agent plan 拒绝权；
- repair eligibility、交互状态和最终写回权。

Router、retrieval、policy、render、scalar repair、batch 和 file applier 都是确定性
组件，不应包装成新 Agent。

## 6. 不可放宽的安全不变量

### 6.1 Miner

- claim 必须是模型所见文本的单一连续 exact substring；
- 保持大小写、空白、标点、数字、日期、版本、identifier、否定、单位和比较词；
- strict Pydantic，`additionalProperties=false`；
- 只有 schema error 可进行既有一次 recovery；
- scope、guard、order、overlap、transport、budget 不重试；
- 无递归 re-windowing 和 while-until-success loop；
- 一个 window/claim/file 失败不能删除其他安全结果。

### 6.2 Evidence

- citation 必须 safe-fetch；
- Web、本地 reference 都是 untrusted data，不是 Agent 指令；
- Judge 只看 bounded evidence；
- evidence 必须是抓取/本地 source 的 exact span；
- snippet 永不作为 evidence；
- SSRF、DNS、redirect、MIME、size、timeout 边界保留；
- 不保存完整网页、raw prompt/response/reasoning/header/key。

### 6.3 修改

- `check` 永不写；
- Agent 永不直接写；
- non-TTY 永不写；
- `fix` 只修改用户逐条批准且通过全部安全门的候选；
- 永不 stage、commit、push；
- stale/overlap/Agent error/无 evidence 都不能强制 apply。

## 7. 已实现能力

当前 Mac working tree 已包含或经报告验证：

- Markdown/README/ADR/Git diff parsing；
- citation、line mapping、trusted source map；
- `live-miner-draft-v3`；
- source-mapped bounded Miner windows；
- `miner-window-policy-v1`；
- `miner-coverage-policy-v2`；
- exact/protected/order/overlap validation；
- typed Coordinator、Scout、Judge、Challenger；
- Deterministic Controller 和 Agent plan validation；
- per-window/per-claim failure isolation；
- safe fetch、HTML extraction、lexical retrieval；
- canonical audit JSON/Markdown、terminal、SARIF；
- ET2001 mining warning；
- ET2002 operational warning；
- evidence-gated candidate patch；
- deterministic offline demo；
- OpenAI-compatible DeepSeek live path；
- Tavily adapter 与真实 Tavily product path；
- 新环境安装/help/demo/check-help smoke。

旧 2026-07-25 handoff 所写的“Coordinator、Scout、Challenger、Web search、
per-claim isolation 尚未实现”已经过时。

这里的 patch 基线只是单目标、只生成不应用的 generic evidence gate/render path；
尚未真实产生非空 contradiction patch，也没有多文件 scalar planner、human
resolution 或 file applier。不要重复实现已有 gate，也不要误认为 `fix` 已存在。

## 8. 精简历史时间线

### Phase 3 evaluation

项目曾过度投入 formal eval。历史事实保留：

- v3_zh 的指标只作历史；
- v4_zh formal gate 未完成，不能写成 passed；
- Phase 3=`completed_with_known_limitations`；
- `phase4_eligible=false`；
- consumed v2/v3/v3_zh/v4_zh 禁止重跑；
- 不创建 v5、新 holdout 或 eval phase。

### 2026-07-27 vertical slice

Checkpoint：

```text
eca904ebdd77c78a838c18b24ba5be63e303e133
```

完成受控 Coordinator、Scout、Judge、Challenger、Controller、trace、failure
isolation、SARIF 和 patch-only product path。后续大量功能仍在 dirty worktree，
因此 HEAD 本身不能代表当前代码。

### Round 1 dogfood

三个固定外部案例均安全结束但为 partial。初始 17 次 Miner dispatch 中 11 次失败，
真实用户完整审计不可靠。

### Miner remediation

- parser 保留 inline code；
- Miner model-visible contract 升至 v3；
- source-mapped bounded windows 将失败 blast radius 降低；
- README/ADR/PR 最终保留 4/6/4 claims，PR 从 0 恢复到 4；
- ET2001 暴露不完整段落；
- coverage v2 的 inline-code identifier signal只增加 visibility，不生成 claim，
  不证明 recall 提升。

Miner P0 仍为 `improved_but_blocking`。

### 2026-07-28 self-use closure

- 增加 canonical operational outcome、ET2002 和更严格 patch evidence gate；
- 第一次 self-use run：`20260728T051838Z-034290e8`，citation 成功抓取却没有
  evidence span；
- 根因是 HTML noise regex 把属性中的任意 `nav` 子串当导航区域；
- `retrieval/extract.py` 使用 token-boundary noise matching 修复；
- 冻结页 0 chunks → 32 chunks；
- 新环境 smoke 通过；
- 真实 Tavily path 验证完成。

## 9. 最新真实运行

唯一补救运行：

```text
run_id: 20260728T065626Z-40dc78f9
exit_code: 2
status: partial
```

Agent logical dispatch：

| Agent | Count |
|---|---:|
| Miner | 9 |
| Coordinator | 2 |
| Scout | 2 |
| Judge | 4 |
| Challenger | 4 |

External activity：

- Tavily HTTP query：2；
- total safe fetch：6；
- 至少一个 Tavily candidate 经 safe fetch 后进入 Judge；
- 三个 cited claims 均进入 Judge；
- `evidence_not_found` 消失。

Claim outcomes：

| Claim | Outcome |
|---|---|
| c1 | challenger_error |
| c2 | challenger_error |
| c3 | judge_error |
| c4 | scout_error |
| c5 | not_in_source |
| c6 opinion | not_checkable |

结论：

- citation handoff=`repaired`；
- Tavily live path=`verified`；
- not-checkable path=`verified`；
- canonical contradiction 未形成；
- candidate patch 为空且未真实验证；
- 最新 run 的 document status=`partial`。

状态不要混为一谈：

```text
self_use_check=operational_partial # 可安全运行，但 downstream 结果不完整
latest_run_document_status=partial
interactive_fix=accepted
candidate_patch=unverified
overall_new_self_use_goal=incomplete
```

最新报告的验证：focused 71、boundary-compliant 605（排除 7 consumed loaders）、
Ruff/mypy/compileall/diff/privacy/demo/new-env smoke 通过。测试数是 checkpoint，不是
产品进度指标。

## 10. 当前未完成和不可误报

未完成：

- downstream Judge/Challenger/Scout 可靠性诊断；
- per-claim safe-fetch/Challenger 上限核对；
- canonical scalar contradiction；
- 非空 evidence-grounded patch；
- 多 targets；
- 多 local MD/TXT references；
- deterministic scalar repair planner；
- interactive `fix`；
- atomic write、applied/reverse diff；
- 真实逐条批准写回。

因此当前可以说“真实 Tavily 与 citation evidence 的多 Agent 审计路径已运行并安全
返回 partial”，不能说“交互式自动修复已经完成”。

## 11. 最新已确认的 `check`/`fix` 产品契约

完整规范见 v2.0 计划。摘要：

- 多 targets、多共享 references；无目录递归；
- 最多 20 targets、50 references、总输入 5 MiB；
- 支持仓库外普通文件；拒绝 symlink；安全 display ID；
- target 不能同时作为 reference；
- citation 和 relevant local refs 都检查；仍 unresolved 才 Tavily；
- citation 或 local reference 任一明确可独立支持候选；
- Tavily 必须由 citation、reference 或 human 额外确认；
- 来源冲突时，`fix` 让 human 从已有 evidence-backed 候选中选择；
- human 不能输入无 evidence 新值；
- Tavily-only 分开询问信任来源和是否 apply；
- 只修整数/小数/百分比、日期 allowlist、SemVer；日期仅接受 `YYYY-MM-DD`、
  `YYYY/MM/DD`、English `Month D, YYYY`、`D Month YYYY`，且 target/evidence 同格式族；
- 不单位换算、不自由改写、不插 citation；
- 正文/列表/表格/inline code 可修；code fence/URL/citation target/front matter/
  HTML attribute 禁止；
- 必须有 final Judge contradiction、成功 Challenger、exact evidence；
- `yes/no/quit`，quit 后再问是否应用之前 yes；
- target/reference SHA、stale checks、per-file atomic write；
- suggested/applied/reverse diff；
- non-TTY 只读；无 `--yes`；
- 非 Git 可用；Git index 不变。

交互顺序必须是：收集 citation/reference/Tavily exact spans → 检测 Tavily-only
或冲突 → human 信任具体 Tavily span/选择 evidence-backed source → Judge 只看
选定 bounded evidence → Coordinator review → 最多一次 Challenger →
deterministic candidate → 单独询问是否 apply。Human 只是 evidence trust/selection
gate，不能覆盖 verdict 或创造新 evidence。

只有 Challenger `uphold` contradiction，或 typed `revise` 后 canonical final
outcome 仍为 `contradicted` 且同一 evidence 精确支持同一 replacement，才可继续。
`abstain`、error 或未完成 contradiction 一律不能修复。

`needs_human` 的准确含义：

- 在 `check` 中是未解决的安全状态；
- 在 `fix` 中，Tavily-only 或 evidence conflict 可通过受限交互解决；
- 无 evidence、Agent error、abstain、stale、禁止位置和 overlap 不能用普通 yes
  绕过。

## 12. 唯一下一任务

不是继续 Miner 调优、扩大 dogfood、做发布或再建 eval。下一任务是执行 v2.0 计划：

1. 先离线分析 frozen run 的 c1/c2 challenger_error、c3 judge_error、c4
   scout_error；
2. 建 per-claim query/URL/fetch/Challenger 表，确认既有预算；
3. 只修 confirmed deterministic downstream bug，不调 prompt/token/retry/budget；
4. 实现多目标 batch 和本地 MD/TXT reference；
5. 实现 deterministic scalar repair eligibility；
6. 实现交互 `fix`、human resolution、atomic write、three diffs；
7. deterministic multi-file acceptance；
8. 代码冻结后，使用本 adjustment 额外且仅有的一次授权进行真实临时副本
   acceptance；第二次 live run 必须重新获得授权。

如果真实模型仍失败，保持
`interactive_fix=offline_complete/live_partial`，不得循环调参或伪造成功。

## 13. Git、文件和凭据边界

根据最新用户报告：

- Mac HEAD 仍为 `eca904ebdd77c78a838c18b24ba5be63e303e133`；
- 后续实现位于 dirty worktree；
- staged diff 为空；
- 无 remote；
- 未 commit、未 push。

新窗口必须保护 dirty worktree：禁止 reset、checkout 覆盖、清理未跟踪 docs/tests、
自动 stage/commit/push。修改前记录相关 hashes；完成后报告，不擅自建立 checkpoint。

环境变量：

```text
OPENAI_API_KEY
OPENAI_BASE_URL
EVIDENCETRACE_MODEL=deepseek-v4-flash
TAVILY_API_KEY
```

只检查 configured/missing，不打印、保存或散列 secret。聊天中曾出现过 key，建议用户
轮换后只在启动执行进程的 shell/secret store 中注入。

## 14. 重要文件导航

Mac：

```text
~/Projects/evidencetrace/
  evidencetrace_self_use_mvp_plan_2026-07-28.md
  docs/PROJECT_HANDOFF.md
  docs/miner-architecture-decision.md
  docs/miner-coverage-failure-diagnosis.md
  docs/dogfood/round_1_2026-07-27.md
  examples/self-use/draft.md
  .evidencetrace/runs/20260728T051838Z-034290e8/
  .evidencetrace/runs/20260728T065626Z-40dc78f9/
  src/evidencetrace/
  tests/

~/Projects/multi_agent_researchops_project_plan.md
```

相关生产模块预计包括：

```text
src/evidencetrace/product.py
src/evidencetrace/audit_models.py
src/evidencetrace/models.py
src/evidencetrace/markdown.py
src/evidencetrace/agents/
src/evidencetrace/retrieval/extract.py
src/evidencetrace/render.py
src/evidencetrace/sarif.py
src/evidencetrace/cli.py  # 实际路径须在 Mac 核验
```

不能根据服务器旧 repo 推断最新实现位置。

## 15. 明确过时的旧结论

不得沿用：

- Coordinator/Scout/Challenger/Web search/per-claim isolation 未实现；
- 当前目标是 PyPI、Demo repo、GitHub Action 或 10 案例；
- 当前需要继续 formal eval、holdout 或 Single-Agent gate；
- patch 在任何情况下都不得 apply；
- 产品只需要单目标、citation/Web，不需要 local reference；
- tests、hash 或文档数量等同于用户进度。

仍然有效：

- `check`、Agent、后台、non-TTY 不得写；
- strict/exact/protected/bounded/failure-isolation；
- snippet 非 evidence；
- stage/commit/push 禁止；
- Phase 3/P0/eligibility 历史状态不得改写；
- 不夸大质量。

## 16. 新窗口汇报协议

每次意见和执行报告都必须包含：

1. 原计划依据；
2. 当前实测依据；
3. confirmed/inferred/unknown；
4. 是否需要调整计划；
5. 用户实际能完成什么；
6. remaining limitation；
7. Git/隐私/secret/副作用。

遇到不确定意图先提问，不要新增 Agent、模块、测试矩阵或发布工作来替用户做决定。

## 17. 接手时应复述的四项

### 项目目标

在 1–2 天内完成可自用的多文件、多证据、交互式 scalar fact fix CLI。

### 当前状态

Citation 与 Tavily 路径已通，check 可运行并安全 partial；downstream Agent errors
仍阻塞 contradiction，interactive fix 尚未实现。

### 禁止事项

不调 Miner/prompt/budget，不建 eval，不做发布，不自由改写，不非交互写，不破坏
working tree，不 stage/commit/push。

### 下一任务

先离线诊断 latest run 的 per-claim errors/budgets，然后按 v2.0 计划实现多目标、local
references、deterministic repair 和交互 atomic fix。
