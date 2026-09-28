# EvidenceTrace 当前交接快照

- 版本：2026-07-29
- 用途：迁移到新电脑后，让没有聊天历史的新会话安全接手
- 性质：描述性快照，不是新的规范计划

## 1. 权威边界

规范性权威依次是用户最新明确决定，以及 Projects 根目录的
`evidencetrace_self_use_mvp_plan_2026-07-28.md` v2.0。

实现事实以当前 Mac 代码和 canonical artifacts 为准。本文件只负责把它们索引到
一起；如果与代码、artifacts 或用户决定冲突，本文件让位。

仓库内同名 self-use 计划是较早的 pre-v2 provenance 副本。旧 dated handoff
`evidencetrace_project_handoff_2026-07-28.md` 同时含有 accepted 状态和实现前的
下一任务，已不适合作为当前入口。

## 2. 项目目标

EvidenceTrace 是一个受 Deterministic Controller 约束的 Multi-Agent
Markdown/TXT 事实审查与交互式修复 CLI：

```text
targets
→ Miner
→ Coordinator
→ citation / local references / authorized Scout
→ bounded exact evidence
→ Judge
→ Coordinator review
→ conditional Challenger
→ deterministic scalar repair
→ human confirmation
→ atomic write + reversible diffs
```

`check` 永远不修改 target、reference 或 Git index，但会写审计 artifacts。`fix`
只允许在交互 TTY 中，对通过证据、Judge、Challenger、位置、类型、stale 和
overlap 安全门的 scalar 候选逐条确认后写回。

## 3. 2026-07-29 实现快照

迁移封装时的 Git 状态：

```text
branch: master
HEAD: eca904ebdd77c78a838c18b24ba5be63e303e133
remote: none
staged: 0
modified: 24
untracked files: 33
```

重要实现位于 dirty worktree，不能只恢复 Git commit。迁移包已保存 `.git`、全部
修改/未跟踪文件、评测资产和 canonical runs。

当前 CLI 已暴露：

```bash
uv run evidencetrace check TARGET1.md TARGET2.txt \
  --reference REF1.md --reference REF2.txt --discover

uv run evidencetrace fix TARGET1.md TARGET2.txt \
  --reference REF1.md --reference REF2.txt --discover
```

Projects 根 v2.0 计划记录的状态字段必须按不同层次理解：

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

`interactive_fix=accepted` 不等于 production-ready、完整事实审计、高 recall，或
Multi-Agent 已证明优于 Single Agent。其他 partial/known-limitations 状态不得被它
覆盖。

## 4. Canonical artifacts

迁移包内保存了两个现有 canonical run：

```text
.evidencetrace/runs/20260728T051838Z-034290e8/
.evidencetrace/runs/20260728T065626Z-40dc78f9/
```

第二个 run 是一次真实只读检查，document status 为 `partial`；它保留 citation、
Tavily、not-checkable 和 downstream Agent failure 的事实。不得因为之后的
interactive fix acceptance 而改写这次历史运行。

## 5. 禁止事项

- 不对 consumed eval loaders 或历史 live runs 重跑取最好结果。
- 不修改 Agent prompt、model、token、retry 或预算来重复调参。
- 不把 Tavily snippet 当 evidence，不绕过 exact span 或 typed verdict。
- 不允许 `check`、non-TTY、Agent、后台或 CI 写目标文件。
- 不把普通 `yes` 当成 Agent error、abstain、stale 或禁止位置的绕过。
- 不 reset、checkout 覆盖、clean、stage、commit、push 或添加 remote。
- 不打印、保存、散列 API key，也不从旧 shell history 恢复 key。
- 已授权的真实 `fix` acceptance 已使用；再次真实 live acceptance 必须重新获得
  用户授权。

## 6. 新电脑上的唯一立即任务

迁移后的第一项工作只有环境恢复和只读一致性检查：

1. 按迁移包 `README.md` 校验 SHA-256 并恢复。
2. 安装锁定依赖，重新输入新的 API key。
3. 只运行 lock check、help、Git 状态、canonical artifact 存在性和
   `evidencetrace demo` 离线 smoke；不要直接运行会包含 consumed loaders 的全量
   pytest。
4. 复述项目目标、状态、禁止事项以及本次用户新指定的任务。

本迁移快照不自动授权新的功能开发或第二次 live acceptance。只读恢复确认完成后，
下一项产品任务由用户另行指定。

## 7. 继续阅读

- [文档索引](README.md)
- [面试前必读指南](interview-guide.md)
- [组件深挖](component-deep-dive.md)
- [工程决策与取舍](engineering-decisions.md)
- [真实问题与工程复盘](problems-and-lessons.md)
- [架构与安全边界](architecture.md)
- [环境配置与迁移恢复](setup-and-recovery.md)
- [已知局限](known-limitations.md)
- `../../evidencetrace_self_use_mvp_plan_2026-07-28.md`：Projects 根规范计划
