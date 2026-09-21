# 项目结构与文件放置约定

本项目同时包含可复现的量化研究、不可变预测服务和若干已在使用的命令行入口。目录整理遵循一个前提：**不能因为重命名或移动文件，改变已发布预测的代码指纹、历史归档路径或用户已有命令。**

当前采用一个正式模型加三个影子模型的统一运行结构。固定计算、数据接入、模型版本、批次发布和管理员对比分别由独立模块负责；职责和维护命令见 [MODEL_PORTFOLIO.md](MODEL_PORTFOLIO.md)。

```text
指数涨跌预测/
|- prediction_service/       FastAPI 服务、SQLite 账本、归档校验、管理面板
|- market_data/              可复现行情输入与 merged_features.csv
|- artifacts/
|  |- baseline/              已冻结的默认模型基线
|  |- evaluation/            候选模型的验收结果
|  |- research/              研究过程与固定研究证据
|  |- releases/              已选择模型的冻结运行包与一致性证据
|  |- run/                   本地命令输出，Git 忽略
|  `- tmp/                   临时实验输出，Git 忽略
|- scripts/                  可执行入口（按职责组织）
|  |- predict.py             本地次日预测
|  |- loop_validation.py     历史循环回放
|  |- shadow_predict.py      固定因果 BiLSTM 影子预测
|  |- shadow_validation.py   固定因果 BiLSTM 影子回放
|  |- fetch_akshare.py       AkShare 行情采集
|  |- fetch_wind.py          Wind 行情采集
|  |- fetch_tushare.py       Tushare 数据采集入口
|  `- run_service.py         预测服务启动入口
|- tools/                    基线冻结、候选评估和研究复核工具
|- tests/                    单元、回归、服务和验收测试
|- docs/                     项目约定、部署和维护说明
|- archive/                  已整理的历史结果、参考资料及哈希清单
|- service_data/             服务运行时数据库、不可变快照和每日归档，Git 忽略
|- config.ini                本机服务配置和凭据，Git 忽略
|- config.ini.example        可提交的无凭据配置模板
|- 循环验证脚本.py            冻结的核心因果回放与预测源码
|- 数据拉取脚本_tushare.py    冻结的 Tushare 数据接入源码
|- tushare_prediction_pipeline.py
|                            冻结的拉取、回放与次日预测管线
|- regularized_direction.py  冻结的方向候选算法
`- return_calibration.py     冻结的收益校准算法
```

## 各目录职责

### `prediction_service/`

仅包含服务实现：配置解析、SQLite 不可变账本、归档哈希校验、调度器、鉴权和 Web 页面。服务的本地状态必须写入 `config.ini` 中指定的 `数据目录`，默认是 Git 忽略的 `service_data/`。

### `market_data/`

这里保存模型可复现所需的输入数据，而不是预测输出：

- `000001_sh.csv`：上证指数原始行情；
- `hangseng.csv`：恒生指数原始行情；
- `merged_features.csv`：供循环验证、次日预测和冻结验收使用的合并特征。

服务刷新时只允许追加新的交易日；已冻结交易日的特征若发生变化，会被服务拒绝。不要手工覆盖历史行，也不要把验证结果写入该目录。

### `artifacts/`

`baseline/`、`evaluation/` 和 `research/` 保存可以追溯的研究证据。验收工具拒绝覆盖已有输出目录，因此一个已有验收目录应被视为冻结记录。

日常命令生成的 CSV、临时比对、探针和本机日志应使用 `artifacts/run/` 或 `artifacts/tmp/`。这两个目录被 Git 忽略，避免测试或临时研究污染冻结证据。

### `tools/` 与 `tests/`

可复用的离线维护命令放入 `tools/`；行为校验放入 `tests/`。新增算法前先在 `tools/` 完成冻结期评估，再把回归断言补入 `tests/`，不要将试验代码直接写入服务目录。

强制连续看涨保护及其发布入口已于 2026-09-19 撤回。相关源码、撤回原因与恢复
验证保存在 `artifacts/evaluation/bullish_guard_withdrawal_20260919/`，不参与正式计算。

### `scripts/` 与根目录源码

`scripts/` 只放用户直接运行的入口，并统一使用模块形式调用：

```powershell
python -m scripts.predict
python -m scripts.shadow_validation
python -m scripts.run_service
```

根目录的 `循环验证脚本.py`、`return_calibration.py`、`regularized_direction.py`、`tushare_prediction_pipeline.py` 和 `数据拉取脚本_tushare.py` 是当前冻结模型合同的一部分。它们不是用户入口目录，工具和服务会直接导入这些源码；新增可执行功能应放在 `scripts/`，不要再把入口放回根目录。

## 历史文件与临时文件

已整理的历史预测 CSV、论文 PDF 和 ZIP 位于 `archive/`，原始路径与 SHA-256 见 `archive/manifest.json`。新的人工导出建议写入 `artifacts/run/`，文件名包含日期或用途，例如：

```text
artifacts/run/20260917_validation.csv
artifacts/run/20260917_next_day_prediction.csv
```

`scripts.predict` 未传入 `--output` 时写入 `artifacts/run/next_day_prediction.csv`；其他研究和验证命令也应显式使用 `artifacts/run/` 或 `artifacts/tmp/`。缓存、pytest 临时目录和被忽略的测试产物可以删除并重新生成。
