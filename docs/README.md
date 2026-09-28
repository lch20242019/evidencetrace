# EvidenceTrace 文档

本目录保存 EvidenceTrace 的当前项目说明、交接快照、架构决策和历史评测记录。
新电脑恢复或新会话接手时，先按下列权威顺序阅读，避免把历史文档误当成当前规范。

## 权威顺序

1. 用户最新明确决定。
2. Projects 根目录的
   `evidencetrace_self_use_mvp_plan_2026-07-28.md` v2.0，它是规范性权威。
3. Mac 当前代码和 canonical artifacts，它们是实现与实测事实权威。
4. [当前交接快照](PROJECT_HANDOFF.md)。
5. 其他 ADR、评测和历史 handoff。

仓库根目录内同名的 `evidencetrace_self_use_mvp_plan_2026-07-28.md` 是较早的
pre-v2 副本，必须保留用于 provenance，但不能覆盖 Projects 根目录的 v2.0 计划。

## 当前项目文档

- [面试前必读指南](interview-guide.md)：项目目标、端到端流程、30 秒/2 分钟/5 分钟
  讲法、常见追问和禁止夸大的表述。
- [组件深挖](component-deep-dive.md)：逐部分说明作用、使用手段、选择原因、问题边界
  和代码入口。
- [工程决策与取舍](engineering-decisions.md)：面试版决策总结，说明 Controller、
  source map、retrieval、provider contract、repair 与评测冻结的理由。
- [真实问题与工程复盘](problems-and-lessons.md)：按现象、根因、处理、验证和剩余边界
  复盘真实故障。
- [当前交接快照](PROJECT_HANDOFF.md)：2026-07-29 的状态、禁止事项和新电脑接续
  步骤。
- [架构与安全边界](architecture.md)：Controller、五个 Agent、证据路由、
  `check`/`fix` 和 artifacts。
- [环境配置与迁移恢复](setup-and-recovery.md)：`uv`、新 API key、迁移包和只读
  验证。
- [已知局限](known-limitations.md)：允许的项目表述、当前 partial 状态和不能夸大的
  能力。
- [实现状态台账](implementation_status.md)：按阶段累计的工程与评测记录。
- [设计决策](design-decisions.md)：append-only 历史决策记录，不替代当前面试总结。

## 产品与架构材料

- [v0.1 产品纵切](v0.1-product-vertical-slice.md)
- [完整文档 benchmark 设计](full-document-benchmark-design.md)
- [代码预算审查](code-budget-review.md)
- [Miner 架构决策](miner-architecture-decision.md)
- [Miner coverage 失败诊断](miner-coverage-failure-diagnosis.md)
- [Phase 3G 架构决策](phase3g-architecture-decision.md)
- [Phase 4B live dev](phase4b-live-dev.md)
- [Phase 4B provider contract](phase4b-provider-contract.md)

这些材料中的历史产品边界可能已被 v2.0 自用计划调整。例如“候选 patch 永不
apply”仍适用于 `check`、非交互、Agent、后台和 CI，但不适用于通过全部安全门且由
用户逐条批准的交互式 `fix`。

## 历史与取证材料

- `evidencetrace_project_handoff_2026-07-28.md`：旧 dated handoff；内部仍有实现前
  的下一任务描述，只作历史。
- `evidencetrace_handoff.md`：更早交接材料。
- `interview-notes.md`：2026-07-10 的 Phase 0 初始面试笔记；当前面试入口已改为
  `interview-guide.md`。
- `dogfood/round_1_2026-07-27.md`：第一轮自用记录。
- `phase3g-v3-zh-postmortem.md`：历史评测复盘。
- `../eval_runs/`、`../eval_sets/`：评测输入和结果，不应擅自重新消费。
- `../.evidencetrace/runs/`：canonical product artifacts。

## 新电脑入口

迁移时先阅读迁移包的 `README.md`，使用 `verify-and-restore.sh` 恢复，再按
[环境配置与迁移恢复](setup-and-recovery.md) 重新安装依赖并输入新的 key。
