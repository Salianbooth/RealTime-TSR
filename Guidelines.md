# 开发规范

以下文档为实时交通标志检测系统项目的完整开发规范，包含目录结构、分支策略、代码风格、提交流程、CI 配置、测试策略等内容。

------

## 1. 仓库目录结构

```text
/
├── src/                       # 源代码目录
│   ├── preprocessing/         # 图像预处理模块
│   ├── detection/             # 交通标志检测模块
│   ├── recognition/           # 交通标志识别模块
│   ├── ui/                    # 用户界面代码
│   └── utils/                 # 通用工具函数
├── data/                      # 数据集及标注（不提交大文件）
│   ├── raw/                   # 原始视频与图像
│   ├── annotations/           # 标注文件（XML/JSON）
│   └── scripts/               # 数据下载/预处理脚本
├── models/                    # 模型权重与导出文件
├── tests/                     # 单元测试与集成测试
├── docs/                      # 项目文档与设计文档
├── .github/
│   ├── workflows/             # GitHub Actions CI 配置
│   ├── ISSUE_TEMPLATE/        # Issue 模板
│   └── PULL_REQUEST_TEMPLATE.md  # PR 模板
├── .gitignore                 # 忽略文件列表
├── README.md                  # 项目简介与快速开始
├── CONTRIBUTING.md            # 贡献指南
├── CODE_OF_CONDUCT.md         # 行为规范
├── LICENSE                    # 开源协议（MIT）
└── setup.py                   # 安装脚本
```

------

## 2. 分支策略

- `main`: 稳定版本，仅合并经 CI 验证和 Review 通过的代码。
- `dev`: 开发主线，日常合入 feature 分支。
- `feature/<name>`: 功能分支，从 `dev` 创建，完成后发 PR 至 `dev`。
- `hotfix/<name>`: 紧急修复分支，直接从 `main` 创建，修复后合并至 `main` 和 `dev`。

> 合并方式统一使用 “Squash and merge”。

------

## 3. Commit Message 规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

- **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- **scope**: 影响范围，如 `(preprocessing)`, `(ui)` 等
- **description**: 简短说明

示例：

```
feat(detection): 添加基于 YOLOv5 的标志检测模块
fix(preprocessing): 修复 CLAHE 参数配置错误
docs: 更新部署文档与使用示例
```

------

## 4. Issue & PR 模板

### Issue 模板（`.github/ISSUE_TEMPLATE/bug_report.md`）

```markdown
---
name: Bug 报告
about: 报告代码 Bug 或异常行为
---

**描述**
简要说明问题

**复现步骤**
1. ...
2. ...
3. ...

**环境**
- OS: Windows/Linux/macOS
- Python: 3.X
- 依赖版本: `opencv-python X.X`, `torch X.X`

**日志/截图**
```

错误日志或截图贴在这里

```

```

### Pull Request 模板（`.github/PULL_REQUEST_TEMPLATE.md`）

```markdown
## 相关 Issue
<!-- 关联的 Issue 编号 -->

## 本次变更
- feat: ...
- fix: ...

## 验证方式
- 单元测试：`pytest tests/`
- 手动测试：说明如何操作

## 备注
其他信息，如性能对比、截图
```

------

## 5. 代码风格与检查

- **Python**: 使用 `black` 进行格式化，`flake8` 做静态检查。
- **配置文件**:
  - `pyproject.toml` / `setup.cfg`：黑格式化、Lint 配置
  - `.flake8`：忽略特定警告
- **Git Hooks**: 安装 `pre-commit`，在提交前自动格式化与检查。

------

## 6. 持续集成 (CI)

在 `.github/workflows/ci.yml`：

```yaml
name: CI
on: [push, pull_request]
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Lint
        run: flake8 src/ tests/
      - name: Format check
        run: black --check src/ tests/
      - name: Run tests
        run: pytest --maxfail=1 --disable-warnings -q
```

- 在 `README.md` 顶部添加 CI 状态徽章。

------

## 7. 测试策略

- **单元测试**：覆盖关键函数（预处理、后处理、分类器）。
- **集成测试**：使用小样本视频，验证端到端流程结果。
- **测试工具**：`pytest`，测试报告输出到 `tests/report/`。

------

## 8. 文档撰写

- `docs/architecture.md`：整体架构与模块图
- `docs/api.md`：各模块 API 使用说明
- `docs/usage.md`：快速开始与示例
- **生成方式**：可选使用 Sphinx 或 MkDocs 生成静态文档

------

## 9. 代码 Review & 合作

- 每个 PR 至少两人 Review，确认功能、性能、可读性。
- 重要逻辑可使用 Pair Programming 或者共享屏幕进行实时讨论。
- Review 后由 PR 发起人或项目负责人合并。

------

## 10. 发布与版本管理

- **版本号**: 遵循语义化版本：`MAJOR.MINOR.PATCH`
- **Release**: 在 GitHub Releases 页面编写 Release Notes，附带二进制或 Docker 镜像。

