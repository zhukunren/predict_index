# 上证指数次日方向预测

本项目在信号日收盘后，使用截至该日可得的上证行情和滞后外部市场特征，预测下一交易日的收盘方向。默认算法是经过冻结历史验收的 `rank_weighted_state_veto_v1`，而不是未通过验收的研究模型。

## 默认算法

默认方向引擎为 `state_veto_rule`：它从滚动历史中选择规则、按历史边际表现赋予排名权重，并在低波动状态中只使用先前完成预测的表现决定是否反向。互补规则若在当前校准期的预测轨迹相同，会合并成一个专家并保留累计的排名权重；因此不会将相同信号描述为独立投票。

方向、幅度和置信度分开处理：

- `predicted_label` 是独立保存的上涨/下跌方向，不会因为幅度收缩为零而改变。
- 涨跌幅使用此前已完成预测的加权中位数缩放，只缩小数值预测，不改变方向。
- 置信度采用固定 300 日、仅使用此前完成结果的 Platt 校准；不会在当前评估期挑选 120、300、600 等窗口。
- 控制台中的“方向正确概率”表示该方向预计正确的概率；“换算上涨概率”由该值和预测方向换算而来。

`regularized_trees` 保留为研究候选。它在开发期表现可接受，但没有通过独立测试期的方向门槛，因此不会替换默认引擎。

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
$env:TUSHARE_TOKEN = "你的 Tushare token"
python tushare_prediction_pipeline.py --validation-days 252 --output artifacts/run/tushare_validation_prediction.csv
```

其中 `--validation-days` 控制 CSV 中的已完成循环验证交易日数。结果首列 `结果类型` 会标记前面的历史行是 `循环验证`，最后一行是 `次日预测`；该行的实际涨跌幅和正确性会留空。默认不保存中间行情文件；传入 `--save-market-data` 可保存到 `market_data`，之后可加 `--skip-fetch` 直接复算。

采集数据可使用 `data_akshare.py`、`数据拉取脚本_tushare.py` 或 `数据拉取脚本_wind.py`。三者都生成 `market_data/merged_features.csv`；预测默认优先读取该文件。

## 预测服务

`prediction_service` 将现有计算流程封装为 FastAPI 服务，包含无需令牌的公开 CSV 接口与账号密码保护的管理员面板。

首次运行前安装服务依赖，并在服务端设置管理员密码和 Tushare token：

```powershell
pip install -r requirements-service.txt
$env:PREDICTION_SERVICE_ADMIN_USERNAME = "admin"
$env:PREDICTION_SERVICE_ADMIN_PASSWORD = "设置一个高强度密码"
$env:TUSHARE_TOKEN = "你的 Tushare token"
python run_service.py
```

默认地址为 `http://127.0.0.1:8000`：

- 管理员面板：`/admin/`
- 公开 CSV：`/api/v1/sh000001/latest.csv`
- 健康检查：`/healthz`

管理员密码只在首次创建本地管理员账号时读取，数据库中仅保存 Argon2 哈希。公开 CSV 不会读取或暴露 Tushare token；每次请求都基于当前归档特征快照重算最近 60 条循环验证和 1 条次日预测，并与不可变预测账本及已发布 CSV 哈希核对。校验失败时 API 返回 `503`，而不会返回漂移后的结果。

管理员点击“刷新数据与预测”后，后台任务会从 Tushare 拉取数据、拒绝覆盖既有历史输入、补齐上一条预测的实际结果、生成新预测，并把特征、CSV、原始行情和哈希清单归档到 `service_data/archives`。提供 `TUSHARE_TOKEN` 时，服务默认会在上海时间 `18:15` 自动提交每日刷新；可通过 `PREDICTION_SERVICE_REFRESH_HOUR`、`PREDICTION_SERVICE_REFRESH_MINUTE` 和 `PREDICTION_SERVICE_SCHEDULED_REFRESH` 调整。

定时任务会跳过周末，并通过 SSE 交易日历过滤节假日；若拉取后行情截止日没有增长，任务记录为 `skipped`，不会创建冗余快照。每次公开 API 重算和管理页读取归档前，都会校验归档特征、结果 CSV、原始行情及清单记录的 SHA-256；校验失败时服务 fail-closed。管理面板在带诊断文件的新快照上还会显示 State Veto 的全部/近 20 日触发率、方向准确率变化和等权方向收益增量。

对外部署时，设置 `PREDICTION_SERVICE_HOST=0.0.0.0` 并通过 HTTPS 反向代理暴露服务，同时设置 `PREDICTION_SERVICE_COOKIE_SECURE=1`。生产环境应将 `PREDICTION_SERVICE_HOME` 放到受备份保护的持久化磁盘，并保留 `service_data/archives` 的全部版本。

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
