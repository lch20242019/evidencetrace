# EvidenceTrace 环境配置与迁移恢复

本文面向本地开发和新电脑恢复。EvidenceTrace 是 Python CLI，不需要数据库或常驻
服务；联网 Agent 路径需要用户在当前 shell 中重新配置 provider key。

## 1. 前置条件

- macOS 和 Xcode Command Line Tools
- Git
- Python 3.12
- `uv`
- 安装依赖时可访问 Python package registry

## 2. 从源码安装

```bash
cd ~/Projects/evidencetrace
uv sync --frozen --extra dev
```

`uv.lock` 已随源码保存。不要复制旧 Mac 的 `.venv`；它不可移植，且迁移包已正确
排除。

## 3. 在新 shell 输入新 key

```bash
export OPENAI_API_KEY='<new-key>'
export OPENAI_BASE_URL='<new-provider-base-url>'
export EVIDENCETRACE_MODEL='<new-model-name>'

# 只有显式使用 --discover 时才需要：
export TAVILY_API_KEY='<new-optional-key>'
```

迁移包不包含旧 `.env` 或 API key。上述变量只存在于当前 shell；关闭终端后需在新
shell 重新输入，除非用户自行选择其他本地配置方式。不要把真实值提交到 Git、写入
项目文档或复制进迁移包。

## 4. 新电脑恢复

先复制整个 `laptop-migration-2026-07-29` 目录，再执行：

```bash
cd /path/to/laptop-migration-2026-07-29
shasum -a 256 -c SHA256SUMS
./verify-and-restore.sh /Users/NEW_USERNAME/Projects --with-local-data
```

恢复脚本拒绝覆盖已有同名目标。EvidenceTrace 源码包默认包含 `.git`、dirty
worktree、全部有效 Markdown、评测资产和两个 canonical runs；`--with-local-data`
另外恢复本地 SQLite cache。

详细归档边界和校验方式以迁移包自己的 `README.md` 与
`MIGRATION_MANIFEST.md` 为准。

## 5. 只读恢复验收

```bash
cd ~/Projects/evidencetrace

git status --short --branch
git log -1 --oneline
git remote -v

uv lock --check --offline
uv run --frozen evidencetrace --help
uv run --frozen evidencetrace check --help
uv run --frozen evidencetrace fix --help

find .evidencetrace/runs -mindepth 1 -maxdepth 1 -type d
```

2026-07-29 迁移快照的预期状态：

```text
branch: master
HEAD: eca904ebdd77c78a838c18b24ba5be63e303e133
staged: 0
modified: 24
untracked files: 33
remote: none
canonical run directories: 2
```

如果迁移包日后重新生成，应以新 manifest 为准。

## 6. 恢复后的安全边界

- 先完整阅读 Projects 根 v2.0 计划和 `docs/PROJECT_HANDOFF.md`。
- 不运行 consumed eval loaders，不重跑历史 live runs。
- 不在没有新授权时执行第二次真实 `fix` acceptance。
- 不使用真实用户文件验证写回；任何后续授权验收仍只能针对临时副本。
- 不 reset、clean、stage、commit、push 或添加 remote。
- 只读检查 key 的 configured/missing 状态，不打印值。

## 7. 常见恢复问题

### `uv` 尝试读取旧缓存

迁移不依赖旧缓存。直接使用新电脑默认 cache；受限环境中可把 cache 指向临时目录：

```bash
UV_CACHE_DIR=/tmp/evidencetrace-uv-cache uv lock --check --offline
```

### 没有 API key

Help 和确定性离线路径仍可运行。模型路径没有 `OPENAI_API_KEY` 时必须安全降级；
`--discover` 没有 `TAVILY_API_KEY` 时应报告 discovery unavailable，不得伪造搜索。

### Git 显示 dirty

这是预期状态，不是恢复失败。重要实现位于未提交和未跟踪文件中；先与
`MIGRATION_MANIFEST.md` 的计数比较，不要清理。
