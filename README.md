# 上证指数次日方向预测

本项目在信号日收盘后，使用截至该日可得的上证行情和滞后外部市场特征，预测下一交易日的收盘方向。默认算法是经过冻结历史验收的 `rank_weighted_state_veto_v1`，而不是未通过验收的研究模型。

## 项目结构

项目按“可复现输入、冻结证据、运行时状态、可执行入口”分层。完整目录说明见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)：

- `prediction_service/`：FastAPI 服务、不可变预测账本和管理面板。
- `market_data/`：可复现的原始行情与合并特征输入。
- `artifacts/`：冻结基线、验收结果与研究证据；临时命令输出统一写入被忽略的 `artifacts/run/`。
- `tools/`：冻结基线与候选验收工具；`tests/`：回归与服务测试。
- 根目录：保留现有中英文可执行入口和算法模块，以兼容既有自动化、历史发布指纹及用户命令。

## 默认算法

默认方向引擎为 `state_veto_rule`：它从滚动历史中选择规则、按历史边际表现赋予排名权重，并在低波动状态中只使用先前完成预测的表现决定是否反向。互补规则若在当前校准期的预测轨迹相同，会合并成一个专家并保留累计的排名权重；因此不会将相同信号描述为独立投票。

方向、幅度和置信度分开处理：

- `predicted_label` 是独立保存的上涨/下跌方向，不会因为幅度收缩为零而改变。
- 涨跌幅使用此前已完成预测的加权中位数缩放，只缩小数值预测，不改变方向。
- 置信度采用固定 300 日、仅使用此前完成结果的 Platt 校准；不会在当前评估期挑选 120、300、600 等窗口。
- 控制台中的“方向正确概率”表示该方向预计正确的概率；“换算上涨概率”由该值和预测方向换算而来。

`regularized_trees` 保留为研究候选。它在开发期表现可接受，但没有通过独立测试期的方向门槛，因此不会替换默认引擎。

## BiLSTM 研究分支

`research/bilstm-causal-evaluation` 分支新增了实验性 `bilstm_causal` 引擎。它以固定 5 个交易日为重训间隔：每个重训点只使用该日之前已完成的标签训练，区间内各交易日复用该因果模型。实时预测和历史回放使用相同的绝对交易日重训锚点。

完整候选运行和 5 个独立实时前缀探针已冻结在 `artifacts/evaluation/bilstm_causal_v1`。探针日期为 `20250102`、`20250610`、`20251110`、`20260415`、`20260911`；预测方向、收益、收盘价、原始置信度和校准置信度均与回放逐项一致。

最终生产口径比较位于 `artifacts/evaluation/bilstm_causal_v3`：它使用 Git 固定输入 `b1a2929:market_data/merged_features.csv`（SHA-256 `11679c0c...cafd2`）及项目既有冻结测试期 `2025-01-02` 至 `2026-09-11`，共 412 个信号日。`v3` 复核了 `v1` 候选输入、模型配置、相关参数、预测文件和实时一致性哈希后，使用生产 CLI 默认参数重新计算 `state_veto_rule` 对照。

| 指标 | `state_veto_rule` | `bilstm_causal` |
|---|---:|---:|
| 方向准确率 | 58.01% | 58.01% |
| 平衡准确率 | 56.66% | 58.15% |
| 固定因果置信度 Brier | 0.24407 | 0.24567 |
| 涨跌幅 MAE | 0.006162 | 0.006211 |
| 涨跌幅 RMSE | 0.008936 | 0.008942 |
| 月度准确率标准差 | 0.10971 | 0.11179 |
| 预测方向切换率 | 41.85% | 8.27% |

BiLSTM 在平衡准确率和方向平滑性上更好，但置信度 Brier、MAE、RMSE 及部分滚动稳定性指标较差；该结果不触发自动替换默认引擎。评估器拒绝覆盖已有输出目录：

```powershell
python tools/evaluate_bilstm_causal.py --output artifacts/evaluation/new_bilstm_comparison
```

研究模式的逐日回放可使用英文兼容入口运行；它不会改变服务默认引擎：

```powershell
python loop_validation.py market_data/merged_features.csv --mode loop_validate --signal-engine bilstm_causal --bilstm-refit-interval 5 --periods 60 --output artifacts/run/bilstm_causal_validation.csv
```

## 运行

使用本地合并特征预测下一交易日，并将结果写到新的路径：

```powershell
python 预测脚本.py --output artifacts/run/next_day_prediction.csv
```

生成一段历史的逐日回放：

```powershell
python 循环验证脚本.py --mode loop_validate --periods 252 --output artifacts/run/validation.csv
```

一次完成 Tushare 拉取、循环验证和最新次日预测，并只输出一份合并结果 CSV：

```powershell
python tushare_prediction_pipeline.py --token "你的 Tushare token" --validation-days 252 --output artifacts/run/tushare_validation_prediction.csv
```

其中 `--validation-days` 控制 CSV 中的已完成循环验证交易日数。结果首列 `结果类型` 会标记前面的历史行是 `循环验证`，最后一行是 `次日预测`；该行的实际涨跌幅和正确性会留空。默认不保存中间行情文件；传入 `--save-market-data` 可保存到 `market_data`，之后可加 `--skip-fetch` 直接复算。

采集数据可使用 `data_akshare.py`、`数据拉取脚本_tushare.py` 或 `数据拉取脚本_wind.py`。三者都生成 `market_data/merged_features.csv`；预测默认优先读取该文件。

## 预测服务

`prediction_service` 将现有计算流程封装为 FastAPI 服务，包含无需令牌的公开 CSV 接口与账号密码保护的管理员面板。

首次运行前安装服务依赖，然后编辑项目根目录的 `config.ini`。该文件已带中文注释；其中至少填写 `[管理员]` 的 `密码`，需要刷新行情时再填写 `[Tushare]` 的 `令牌`。`config.ini` 被 Git 忽略，方便只在服务器本地保存凭据；从新克隆的仓库首次部署时，先用无凭据模板创建它：

```powershell
pip install -r requirements-service.txt
# 仅当 config.ini 尚不存在时执行
Copy-Item config.ini.example config.ini
python run_service.py
```

默认地址为 `http://127.0.0.1:8000`：

- 管理员面板：`/admin/`
- 公开 CSV：`/api/v1/sh000001/latest.csv`
- 健康检查：`/healthz`

管理员密码只在首次创建本地管理员账号时读取，数据库中仅保存 Argon2 哈希。公开 CSV 不会读取或暴露 Tushare token；每次请求都基于当前归档特征快照重算最近 60 条循环验证和 1 条次日预测，并与不可变预测账本及已发布 CSV 哈希核对。校验失败时 API 返回 `503`，而不会返回漂移后的结果。

管理员点击“刷新数据与预测”后，后台任务会从 Tushare 拉取数据、拒绝覆盖既有历史输入、补齐上一条预测的实际结果、生成新预测，并把特征、CSV、原始行情和哈希清单归档到 `service_data/archives`。填写 `[Tushare]` 的 `令牌` 后，将 `[服务]` 的 `启用定时刷新` 设为“是”，即可在上海时间 `刷新小时:刷新分钟` 自动提交每日刷新。

定时任务会跳过周末，并通过 SSE 交易日历过滤节假日；若拉取后行情截止日没有增长，任务记录为 `skipped`，不会创建冗余快照。每次公开 API 重算和管理页读取归档前，都会校验归档特征、结果 CSV、原始行情及清单记录的 SHA-256；校验失败时服务 fail-closed。管理面板在带诊断文件的新快照上还会显示 State Veto 的全部/近 20 日触发率、方向准确率变化和等权方向收益增量。

默认正式发布始终使用 `state_veto_rule`。若服务器有足够的 CPU 资源，可启用 BiLSTM 影子运行：它会在默认发布成功后，基于同一个不可变行情快照异步计算 `bilstm_causal`，写入独立模型 release、预测账本和 `service_data/shadow_archives`，不会改变公开 CSV、活动 publication 或默认信号。

在 `config.ini` 的 `[BiLSTM影子]` 中将 `启用` 改为“是”，并按需调整 `循环验证天数` 和 `重训间隔交易日` 后重启服务。

管理员面板可手动提交一次影子任务、查看其状态、最近 60 日胜率与平衡准确率，并下载影子 CSV。影子任务失败只记录失败原因，不会影响已发布的默认预测。

对外部署时，在 `config.ini` 中将 `[服务]` 的 `监听地址` 设为 `0.0.0.0`，并通过 HTTPS 反向代理暴露服务；反向代理正确终止 TLS 后，再将 `Cookie仅HTTPS` 设为“是”。生产环境应将 `数据目录` 指向受备份保护的持久化磁盘，并保留其 `archives` 子目录的全部版本。自定义配置路径可使用 `python run_service.py --config D:\secure\prediction.ini`。

为便于 Linux、CI 和英文工具链使用，以下英文入口与原中文脚本等价，原有中文文件及命令保持不变：

- `loop_validation.py` -> `循环验证脚本.py`
- `predictor.py` -> `预测脚本.py`
- `tushare_fetcher.py` -> `数据拉取脚本_tushare.py`

预测引擎源码或固定参数发生变更后，服务不会覆盖旧预测，也不会自动切换版本。只有在当前代码对活动归档重算并与旧账本逐字段一致时，才可执行兼容发布升级：

```powershell
python run_service.py --promote-compatible-release
```

校验失败会停止升级并保留原发布版本。

## 非回退验收

旧版默认算法已冻结在 `artifacts/baseline/default`，包含源码提交、输入哈希、逐日预测和完整配置。默认冠军的最终验收位于 `artifacts/evaluation/default_champion_v3`。

验收命令会拒绝已有输出目录，确保结果不能被覆盖：

```powershell
python -B tools/evaluate_prediction_candidate.py `
  --candidate fixed_state_veto `
  --output artifacts/evaluation/default_champion_v3
```

候选必须与冻结基线使用相同日期和真实收益，并满足：

- 默认后处理版本每个信号日的方向都完全一致。
- 测试期、最近 252 日和最近 20 日的方向准确率、平衡准确率不低于基线。
- 相同窗口的涨跌幅 MAE 和 RMSE 不高于基线。
- 置信度只按共同的固定 300 日因果校准过程比较 Brier 分数。

当前冻结测试期为 2025-01-02 至 2026-09-11，共 412 个信号日。最终冠军与旧版 1,394 个冻结信号日方向完全一致；测试期方向准确率均为 58.01%，同时 MAE 从 0.006916 降到 0.006162，RMSE 从 0.009673 降到 0.008936。历史验收不能保证未来收益，新的方向模型仍需先经过独立开发期和冻结测试期，才能成为默认算法。
