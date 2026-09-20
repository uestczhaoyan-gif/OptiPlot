# 验证记录

## 2026-09-20 · 数据层纯化（N1）

本机环境：Python 3.14.6、NumPy 2.4.6、pandas 3.0.3、Matplotlib 3.11.0、SciPy 1.18.0（Anaconda，通过 `launch.py` 的探测路径使用，未安装或修改任何环境）。

自动测试：**47 项通过 + 14 项子测试**（pytest）。

本轮改动：

- `catalog/cases.json`（108 条带出处案例）→ `catalog/styles.json`（108 条，仅 `id` / `label` / `patterns` / `data_schema` / `recipe`），迁移脚本保留在 `research/migrate_cases_to_styles.py`
- 移除 `app.py` 的"论文图表案例"Tab 及逐图证据展示（该 Tab 是案例库唯一的运行时读者），`app.py` 576 → 473 行
- 移除 `core.py` 中声明后从未被引用的 `recommend(catalog_path=None)` 死参数
- 新增 `render.FIGURE_TYPES` 作为图型 ID 的权威列表
- 新增 `optiplot/__init__.py` 未导出的 `FIGURE_TYPES` 由测试直接引用

本轮测试：

- `test_styles_reference_real_figure_types`：`styles.json` 每条的唯一 ID、四个必需字段非空，且每个 `patterns` 值都对应 `FIGURE_TYPES` 中真实存在的渲染分支
- `test_styles_carry_no_per_figure_provenance`：断言七类出处字段（`doi.org` / `evidence` / `source_url` / `paper_id` / `visual_review` 等）没有回潮
- `test_papers_remain_valid_bibliography`：题录唯一 ID 与完整性；**不再**与 `styles.json` 建立外键
- 原有 45 项核心与渲染测试全部保持通过，未修改任何推荐规则或渲染分支

### 本轮发现的环境不一致（尚未处理）

1. **本机 pandas 3.0.3 超出声明范围。** `pyproject.toml` 写的是 `pandas>=2.0,<3`。测试在 pandas 3 上全绿，说明上界可能过紧，但 pandas 3 引入了 copy-on-write 与字符串 dtype 默认值变更，全绿不等于无行为差异。需要决定：放宽上界并在 CI 覆盖，还是把开发环境钉在 `<3`。
2. **本机 Python 3.14 不在 CI 矩阵内。** CI 跑 3.10 / 3.12 / 3.13，即日常开发所用的解释器版本从未被远端验证过。

### 仍未验证

跨系统 GUI、超大文件性能、MAT v7.3、专业多面板排版。GitHub Actions 已在远端实际运行并通过（见下）。

## 2026-09-20 · 首次推送到 GitHub

`371af52` 推送后 Actions 工作流在 Python 3.10 / 3.12 / 3.13 三个版本上运行，结论 **success**。这是本项目第一次远端验证。

## 2026-09-14 · 0.2.0 初始验证（历史记录）

本机环境：Python 3.13.5、NumPy 2.1.3、pandas 2.2.3、Matplotlib 3.10.0、SciPy 1.15.3。启动器可通过用户已有的 Conda 环境清单找到该解释器，无需安装或修改环境。

自动测试：**45 项通过**（pytest）。

已完成：

- 核心规则测试：空文件、表头、常量/编号、角度单位、有符号响应、重复测量、真实/伪二维网格、类别、MAT/NPY、缺失与无穷。
- 全部 CSV 模拟数据的所有推荐均能生成非空 PNG。
- 独立导出 ZIP 在临时目录执行成功，重新产生 PNG、SVG、PDF。
- 验证不自动拟合、缺失间断不连线、不同组的误差不混算、单点组不虚构 SD。
- GUI 实际创建窗口、加载数据、展示四个候选、切换编辑器、显示导出按钮并退出；窗口截图在 `docs/images`。
- CLI 对光谱数据导出图片与可复现包；产物不检入仓库，由 `python -m optiplot.cli examples/sample_spectrum.csv --out <目录> --bundle` 现场生成。
- 文献 JSON 唯一 ID、必需字段、渲染器名称检查；原图抽查覆盖 3 页、7 条案例。（当时的论文外键与计数一致性校验已在 2026-09-20 随逐图出处字段一并移除。）
- OpenAlex 采集器做过离线分页和去重验证；尚无本次在线 OpenAlex 采集结果。

网络安装曾因自动审批服务配置错误失败。随后使用本机已安装的依赖完成运行验证；项目不依赖那次安装操作。GUI 截图通过应用自身窗口渲染获取，不截取其他窗口。
