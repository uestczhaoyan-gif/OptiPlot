# 文献图表案例库与持续收集

本版是可核查的起始案例库：**25 篇题录、108 条图表/子面板结构案例、15 个期刊或会议名称、9 类绘图模式**。23 篇有具体案例；另 2 篇仅核对题录，明确保留为待补图表的候选。108 是结构案例数，同一张多面板图可贡献多个不同的数据展示问题，**不是 108 张独立下载的原图，也不是 108 篇论文**。

选题覆盖红外光谱、光电探测、偏振与 Stokes、超表面、金属透镜、计算成像和视觉会议中的方法比较。它不是完整的一区二区文献普查，也未经过独立审美评分。结构选择来自已读图注或出版方图题，绘图配方是本项目编写的迁移建议。

## 文件与证据

- `catalog/papers.json`：题录、DOI/一手链接、年份、类型、分区待核验状态及原图授权状态。
- `catalog/cases.json`：论文 ID、准确图号/子图号、展示目的、适用渲染器、数据列模式、实现步骤和证据级别。
- `research/collect_openalex.py`：游标分页、缓存、重试、去重的题录采集器；只生成候选文件，不修改人工案例库。开发期工具，不随产品运行时使用。

|证据级别|案例数|能支持的结论|
|---|---:|---|
|`local_full_text_caption`|90|已读本机现有文章全文中的图注；未逐像素审阅全部原图|
|`publisher_full_text_caption`|5|已读出版方网页的完整图注|
|`primary_search_index_caption`|11|已读一手出版方/作者文档在搜索索引中的图注或图表说明；直接全文访问不一定成功|
|`publisher_figure_title_only`|2|只确认出版方列出的图题，子图细节仍待核对|

另对 3 篇论文各一页代表性原图进行了视觉复核，覆盖 7 条案例，已单独记录 `visual_review`。复核纠正了损耗比曲线误归为热图，以及倒数坐标的变量定义。其余案例仍按上述证据级别使用，不视为已逐图审美审查。

本地来源为用户已有的 V-Lab 文本提取结果。仅在题录中记录文件名，不随开源项目分发这些全文。综述、Perspective 和 News & Views 的案例额外标为二次文献结构：若要复用具体科学结论，应沿原文引用追溯研究文章。

某些配方将原展示改为更适合通用渲染器的形式，例如将多通道结果改为表格，或将装置示意抽象为流程节点。此类记录在配方中说明；`patterns` 表示可迁移的绘图方法，不宣称原文实际使用了每一种备选图形。近场热图也不能仅凭二维数据自动生成真实三维器件结构。

## 分区如何处理

所有期刊的 `quartile` 都初始化为：

```json
{"system": null, "year": null, "value": null, "status": "unverified"}
```

“一区二区”至少需要区分 **JCR 学科分区** 与 **中科院期刊分区**，并明确年份、学科和大类/小类口径。一个期刊可能在不同学科拥有不同分区。会议另用会议评价体系，不应把 CVPR、ICCV 或 SIGGRAPH 直接填成 JCR Q1。

因此当前按主题和来源筛选，不使用未经核验的 Q1/Q2 标签。后续可导入用户有权使用的分区清单，通过 ISSN/期刊 ID 与年份匹配，再人工确认多学科条目；不要根据期刊名、影响因子或网站搜索摘要猜测分区。

## 如何从案例迁移到自己的数据

1. 先确认列对应的物理量、单位、样品/条件、重复测量身份与缺失值。
2. 用 `data_schema` 检查是否具备该图所需的数据；例如误差棒需要真实重复测量或显式不确定度，流程图需要节点和边。
3. 依据 `recipe` 选择坐标、分组和归一化，再在 GUI 中比较可用预览。原始数据不会由论文案例填充。
4. 原图的指标定义、尺度、测试条件和模型选择仍需查阅链接。拟合关系、置信区间或论文结论不能通过模仿外观获得。

配方中特别保留：有符号偏振响应不要直接作为极图半径、分母近零的响应比必须标记无效、不同量纲优先分面、误差棒必须说明 SD/SEM/CI 与 n、图像比较应共享色标、光谱偏移和去卷积/平滑需记录。

## 持续扩充题录

采集器仅依赖 Python 标准库，OpenAlex 负责检索元数据。当前账户和 API key 要求以 [OpenAlex 开发者文档](https://developers.openalex.org/) 为准。可以通过环境变量 `OPENALEX_API_KEY` 提供用户自己的密钥；脚本参数可以省略密钥，但服务端若拒绝匿名请求会清楚报错，不能保证匿名调用可用。

在项目目录执行：

```powershell
python research/collect_openalex.py --query "metasurface photodetector" --query "computational imaging" --year-from 2020 --year-to 2026 --max-pages 5 --per-page 50 --max-records 400
python research/collect_openalex.py --query "polarization detector" --filter "is_oa:true" --max-pages 2 --max-records 100
```

输出默认是 `catalog/openalex_candidates.json`，缓存默认位于 `.cache/openalex/`。限制是上限，实际命中数量以日志和输出统计为准；从未执行过在线收集时，不能将这些上限当作已收集数量。

- `--query` 可以重复，按 DOI/文献 ID 去重，并保留命中的多个检索词。
- `--max-pages` 是每个主题的页数上限，`--max-records` 是全局去重上限。
- `--filter` 可附加官方 API 支持的条件，例如来源 ID；来源 ID 需先核实，脚本不猜期刊。
- 默认尊重缓存；`--refresh` 重新请求，`--offline` 只回放已有缓存。
- 网络失败会有限重试；遇到长时间限流会停止并保留结果。分页成功后立即保存，重新运行可利用已有缓存。
- `--output` 可以指定候选文件，但禁止覆盖人工的 `papers.json` 和 `cases.json`。

取得候选后，仍需要打开合法可访问的出版方页面/已有 PDF，检查具体图注、图号、样本结构和可迁移方法，再人工添加案例。**本版本没有把题录搜索包装成自动全文图表理解，也没有宣称下载过大规模论文图像。**

## 原图、开源与复现边界

项目保存题录链接和原创结构化笔记，不下载或再分发论文原图。开放获取元数据、可阅读 PDF 和原图可再分发是不同权限；尤其综述中的第三方转载图可能另有版权。项目软件许可证不替代原论文图片的授权。

预览由用户数据或清楚标明的示例数据生成。它们是绘图方法预览，不是原论文实验结果，也不承诺像素级复刻。

## 本次验证与限制

已核对全部案例的论文外键、唯一 ID、必需字段、渲染器名称和 JSON 可读性。采集器做了离线缓存分页/去重验证，实际网络采集未作为完成项：本机直接联网请求被沙箱阻止，自动批准审核又因审核服务模型配置错误而拒绝；文献检索改用可用的网页工具完成。本版本未保存 OpenAlex 在线采集结果，也没有因此伪造数据量。

## 题录索引

|论文/年份|期刊或会议|案例数|证据范围|
|---|---|---:|---|
|[Ultrasensitive Transmissive Infrared Spectroscopy via Loss Engineering of Metallic Nanoantennas for Compact Devices](https://doi.org/10.1021/acsami.9b18002) (2019)|ACS Applied Materials & Interfaces|8|本地全文图注|
|[Metamaterial technologies for miniaturized infrared spectroscopy: Light sources, sensors, filters, detectors, and integration](https://doi.org/10.1063/5.0033056) (2020)|Journal of Applied Physics|3|本地全文图注|
|[Zero-bias mid-infrared graphene photodetectors with bulk photoresponse and calibration-free polarization detection](https://www.nature.com/articles/s41467-020-20115-1) (2020)|Nature Communications|11|本地全文图注|
|[Midinfrared semimetal polarization detectors with configurable polarity transition](https://www.nature.com/articles/s41566-021-00819-6) (2021)|Nature Photonics|8|本地全文图注|
|[Geometric filterless photodetectors for mid-infrared spin light](https://www.nature.com/articles/s41566-022-01115-7) (2023)|Nature Photonics|7|本地全文图注|
|[Metaphotonic photodetectors for direct Stokes quantification](https://www.nature.com/articles/s41928-025-01481-4) (2025)|Nature Electronics|6|本地全文图注|
|[Filterless vector light field photodetector based on photonic-electronic co-designed non-Hermitian silicon nanostructures](https://doi.org/10.1364/OE.550582) (2025)|Optics Express|9|本地全文图注|
|[Semiconductive optoelectronic refractive index sensors with on-chip electrical readout](https://doi.org/10.1364/OL.569302) (2025)|Optics Letters|7|本地全文图注|
|[Anomalous diffraction of metagratings near dielectric interfaces](https://doi.org/10.1088/2040-8986/ae4bf0) (2026)|Journal of Optics|5|本地全文图注|
|[Bridging the gap in flat optics: the dawn of partially nonlocal metasurfaces](https://www.nature.com/articles/s41377-026-02397-0) (2026)|Light: Science & Applications|2|本地全文图注|
|[Dispersive optical activity for spectro-polarimetric imaging](https://www.nature.com/articles/s41377-025-01766-5) (2025)|Light: Science & Applications|2|本地全文图注|
|[2D materials-based next-generation multidimensional photodetectors](https://www.nature.com/articles/s41377-025-01995-8) (2025)|Light: Science & Applications|8|本地全文图注|
|[Polarization-Sensitive Photoelectric Conversion](https://doi.org/10.1021/acs.chemrev.5c00693) (2026)|Chemical Reviews|6|本地全文图注|
|[Snapshot computational spectroscopy enabled by deep learning](https://doi.org/10.1515/nanoph-2024-0328) (2024)|Nanophotonics|8|本地全文图注|
|[A monolithic immersion metalens for imaging solid-state quantum emitters](https://www.nature.com/articles/s41467-019-10238-5) (2019)|Nature Communications|5|一手图题/图注|
|[A broadband achromatic metalens for focusing and imaging in the visible](https://www.nature.com/articles/s41565-017-0034-6) (2018)|Nature Nanotechnology|2|一手图题/图注|
|[Inverse design enables large-scale high-performance meta-optics reshaping virtual reality](https://www.nature.com/articles/s41467-022-29973-3) (2022)|Nature Communications|3|一手图题/图注|
|[Bright-field holography: cross-modality deep learning enables snapshot 3D imaging with bright-field contrast using a single hologram](https://www.nature.com/articles/s41377-019-0139-9) (2019)|Light: Science & Applications|0|仅题录，图表待补|
|[Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields](https://openaccess.thecvf.com/content/ICCV2023/papers/Barron_Zip-NeRF_Anti-Aliased_Grid-Based_Neural_Radiance_Fields_ICCV_2023_paper.pdf) (2023)|ICCV|1|一手图题/图注|
|[Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://openaccess.thecvf.com/content/CVPR2022/papers/Barron_Mip-NeRF_360_Unbounded_Anti-Aliased_Neural_Radiance_Fields_CVPR_2022_paper.pdf) (2022)|CVPR|1|一手图题/图注|
|[Deep Image Prior](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf) (2018)|CVPR|1|一手图题/图注|
|[Learning to See in the Dark](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf) (2018)|CVPR|1|一手图题/图注|
|[ADFactory: An Effective Framework for Generalizing Optical Flow with NeRF](https://openaccess.thecvf.com/content/CVPR2024/papers/Ling_ADFactory_An_Effective_Framework_for_Generalizing_Optical_Flow_with_NeRF_CVPR_2024_paper.pdf) (2024)|CVPR|3|一手图题/图注|
|[Neural Microfacet Fields for Inverse Rendering](https://openaccess.thecvf.com/content/ICCV2023/papers/Mai_Neural_Microfacet_Fields_for_Inverse_Rendering_ICCV_2023_paper.pdf) (2023)|ICCV|1|一手图题/图注|
|[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://doi.org/10.1145/3592433) (2023)|ACM Transactions on Graphics / SIGGRAPH|0|仅题录，图表待补|
