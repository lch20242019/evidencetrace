# EvidenceTrace 架构与安全边界

本文描述 2026-07-29 当前工作树中的产品架构。规范性需求仍以 Projects 根目录的
v2.0 self-use 计划为准。

## 1. 系统定位

EvidenceTrace 是本地 CLI，不是通用 truth detector、自动写作助手或无人值守修改
服务。它对一个或多个 Markdown/TXT 目标建立 source-located claims，用受限证据和
typed Agent verdict 完成审计；只有交互式 `fix` 可以在严格门控后修改目标。

## 2. 主链路

```text
explicit targets + shared references
        │
        ▼
input preflight / safe IDs / SHA / output collision checks
        │
        ▼
Miner ──► Coordinator initial plan
        │
        ├── direct citation safe fetch
        ├── bounded local-reference retrieval
        └── authorized Scout/Tavily + safe fetch
        │
        ▼
deterministic evidence routing / conflict handling
        │
        ├── check: unresolved evidence → needs_human
        └── fix: Tavily-only/conflict
                 → human evidence trust/selection
        │
        ▼
Judge ──► Coordinator review ──► conditional Challenger
        │
        ▼
canonical outcome / policy / terminal / JSON / Markdown / SARIF
        │
        └── fix only: scalar planner
                    → per-candidate apply confirmation
                    → stale checks → atomic write
                    → suggested/applied/reverse diff
```

## 3. 权限所有者

Deterministic Controller 不是 Agent。它拥有：

- 文件、网络和写回权限；
- Agent 调度、预算、重试和失败隔离；
- evidence trust/selection 的门控与状态、bounded payload 和 conflict 状态；
- policy、exit code、canonical artifacts；
- repair eligibility、TTY 交互、stale 检查和 atomic write。

具体 evidence trust/selection 由 human 决定，Controller 只约束可选 evidence 和执行
流程。Agent 不能自行读取任意文件、调用任意网络、修改目标、stage、commit 或
push。

## 4. Agent 职责

- **Miner**：提取逐字符一致的 atomic claims、citation 和位置；不判断真假。
- **Coordinator**：产生 typed initial/review plan；不拥有工具权限。
- **Scout**：只为授权的 unresolved claim 发现候选 URL；snippet 不能成为 evidence。
- **Judge**：一次只比较一个 claim 与 bounded evidence。
- **Challenger**：只复核高风险 verdict，每个 claim 最多一次。

任何 claim/window/source 的 Agent 或 transport 失败都形成明确的 partial 状态，不应
终止无关 claim，也不能生成可应用 repair。

## 5. 证据路径

证据按以下路径进入：

1. 直接 citation 经 safe fetch 和 bounded extraction；
2. 用户显式提供的共享 Markdown/TXT references；
3. 只有 `--discover` 明确授权时才使用 Scout/Tavily。

URL、redirect、响应类型、大小和私有/metadata 地址受 safe-fetch 约束。Search
snippet 只用于选择 URL。可交给 Judge 的内容必须是抓取后可定位的 exact bounded
span。

Citation/reference 无冲突时可直接审计。Tavily-only 或来源冲突在 `check` 中保持
`needs_human`；在交互式 `fix` 中，人只能信任具体 evidence span，或从现有
evidence-backed 候选中选择，不能输入任意新值。

## 6. Deterministic repair

第一版 repair 只处理：

- 整数、小数和百分比；
- allowlist 中的日期格式；
- SemVer。

它不自由改写、不换单位、不插 citation、不修改 fenced code、URL、citation target、
front matter 或 HTML attribute。可应用候选必须同时具有 final Judge
`contradicted`、成功 Challenger、exact evidence、同类型/单位 replacement、允许
位置、非重叠 span 和有效 target/reference SHA。

## 7. `check` 与 `fix`

`check` 始终只读，可以输出 canonical audit、Markdown、SARIF 和 suggested diff，
但不修改目标或 Git index。

`fix` 必须运行在交互 TTY 中。每个候选分别进行 evidence trust/selection 和
apply 决策；`quit` 后还要单独确认是否应用之前批准项。写回前重新检查 SHA，在内存
生成 forward/reverse diff 并验证 round trip，然后使用同目录临时文件原子替换。

## 8. Artifacts

单目标历史 run 位于：

```text
.evidencetrace/runs/<run-id>/
```

Batch/fix 的设计结构是：

```text
<run-root>/
  batch-manifest.json
  targets/<safe-target-id>/
    audit.json
    audit.md
    results.sarif
    suggested.diff
    applied.diff
    reverse.diff
```

每目标 `audit.json` 是 canonical truth；其他输出由 canonical outcome 渲染。
Artifacts 不应保存 API key、authorization header、raw reasoning、完整私有文件或
用户主目录绝对路径。

## 9. 主要代码入口

- `src/evidencetrace/cli.py`：CLI 命令和参数。
- `src/evidencetrace/self_use_inputs.py`：多目标/reference preflight。
- `src/evidencetrace/self_use.py`：自用 batch/fix 编排。
- `src/evidencetrace/product.py`：产品主链路和 Controller 行为。
- `src/evidencetrace/local_evidence.py`：本地证据。
- `src/evidencetrace/repair.py`：deterministic repair 与写回安全。
- `src/evidencetrace/agents/`：五个 Agent。
- `src/evidencetrace/retrieval/`：safe fetch、抽取和排序。
- `src/evidencetrace/artifacts.py`、`render.py`、`sarif.py`：canonical 输出。
