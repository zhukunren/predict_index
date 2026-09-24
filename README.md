# 上证指数次日方向预测

本项目在信号日收盘后，预测下一交易日的上证指数方向，并通过公开 API 提供最近 60 个交易日与次日预测的合并 CSV。

当前服务采用 **期权＋资金流组合** 作为正式生产模型，原生产基线、资金流信号、资金流＋价格特征组合作为三个影子模型。四个模型共享行情、独立保存预测，每日上海时间 **18:30** 刷新。管理员在 `/admin/models` 查看同日期统计、逐日预测、方向分歧和事前表现，并可自定义统计交易日数、导出对比 CSV。

此次切换依据用户于 2026-09-19 的明确选择；原研究验收结果保留，不将上线解释为已经证明性能不回退。模型运行结构、冻结参数、数据时点和维护流程见 [docs/MODEL_PORTFOLIO.md](docs/MODEL_PORTFOLIO.md)。

## 项目结构

项目按“可复现输入、冻结证据、运行时状态、可执行入口”分层。完整目录说明见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)：

- `prediction_service/`：FastAPI 服务、不可变预测账本和管理面板。
- `market_data/`：可复现的原始行情与合并特征输入。
- `artifacts/`：冻结基线、验收结果与研究证据；临时命令输出统一写入被忽略的 `artifacts/run/`。
- `scripts/`：预测、数据采集、影子模式和服务启动入口。
- `tools/`：冻结基线与候选验收工具；`tests/`：回归与服务测试。
- 根目录：只保留冻结模型源码、Tushare 预测管线、配置模板和项目文档。

## 原生产基线与命令行默认算法

底层基线方向引擎为 `state_veto_rule`（`rank_weighted_state_veto_v1`）：它从滚动历史中选择规则、按历史边际表现赋予排名权重，并在低波动状态中只使用先前完成预测的表现决定是否反向。互补规则若在当前校准期的预测轨迹相同，会合并成一个专家并保留累计的排名权重；因此不会将相同信号描述为独立投票。原命令行入口保留这一算法；正式组合服务使用独立的固定模型运行包。

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

研究模式的逐日回放可使用 `scripts/` 下的英文入口运行；它不会改变服务默认引擎：

```powershell
python -m scripts.loop_validation market_data/merged_features.csv --mode loop_validate --signal-engine bilstm_causal --bilstm-refit-interval 5 --periods 60 --output artifacts/run/bilstm_causal_validation.csv
```

也可以使用固定影子模式入口，脚本会强制使用 `bilstm_causal`、5 个交易日重训间隔并关闭 recent-failure guard，不能被通用 CLI 参数改成正式引擎：

```powershell
python -m scripts.shadow_validation --periods 60
python -m scripts.shadow_predict --json
```

两个脚本的默认结果分别写入 `artifacts/run/bilstm_causal_validation.csv` 和 `artifacts/run/bilstm_causal_next_day_prediction.csv`。
影子次日预测会先进行因果历史回放与置信度校准，默认会立即显示设备和输入数据，并在每 30 秒输出一次仍在运行的提示；`--verbose` 会额外输出每轮训练明细，`--quiet` 可关闭过程提示。

要验证影子实时预测与历史回放的一致性，可对同一份特征快照运行以下命令。工具会默认选择倒数第二个交易日，让历史回放可见一条未来行情，而独立实时预测只使用该日之前的前缀；报告逐字段比较方向、涨跌幅、收盘价、原始/校准置信度和收益校准系数：

```powershell
python tools/verify_shadow_live_replay.py --input artifacts/tmp/shadow_parity_tushare_20260917/merged_features.csv --output artifacts/run/shadow_live_replay_parity.json --device cuda
```

传入 `--probe-count 3 --probe-window-days 60` 可从最近 60 个可回测交易日中均匀选择 3 个真实日期，并使用一次连续历史回放与三次独立实时前缀预测完成批量验证。

## 运行

使用本地合并特征预测下一交易日，并将结果写到新的路径：

```powershell
python -m scripts.predict --output artifacts/run/next_day_prediction.csv
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

采集数据可使用 `python -m scripts.fetch_akshare`、`python -m scripts.fetch_tushare` 或 `python -m scripts.fetch_wind`。三者都生成 `market_data/merged_features.csv`；预测默认优先读取该文件。

## 预测服务

`prediction_service` 将现有计算流程封装为 FastAPI 服务，包含无需令牌的公开 JSON 接口与账号密码保护的管理员面板。

首次运行前安装服务依赖，然后编辑项目根目录的 `config.ini`。该文件已带中文注释；其中至少填写 `[管理员]` 的 `密码`，需要刷新行情时再填写 `[Tushare]` 的 `令牌`。`config.ini` 被 Git 忽略，方便只在服务器本地保存凭据；从新克隆的仓库首次部署时，先用无凭据模板创建它：

```powershell
pip install -r requirements-service.txt
# 仅当 config.ini 尚不存在时执行
Copy-Item config.ini.example config.ini
python -m scripts.run_service
```

默认地址为 `http://127.0.0.1:8000`：

- 管理员面板：`/admin/`
- 公开 JSON：`/api/v1/sh000001/latest.json`
- 公开统计与方向诊断：`/api/v1/sh000001/metrics?days=60`，支持 1 至 5000 个已结算交易日
- 健康检查：`/healthz`
- 进程存活：`/livez`；公开数据状态：`/api/v1/sh000001/status`

管理员密码只在首次创建本地管理员账号时读取，数据库中仅保存 Argon2 哈希。公开 JSON 从已校验的发布记录生成，每条记录仅包含 `signal_date`、`predicted_next_day_return`、`predicted_direction`、`predicted_next_day_close`、`confidence`、`actual_next_day_return`、`direction_prediction_correct`。日期为 `YYYYMMDD` 整数；涨跌幅与置信度为小数比例；未结算的实际涨跌幅和正确性为 `null`。接口支持 `GET`、`HEAD`、`ETag / If-None-Match` 和 `304`，访问时不会运行预测或请求行情；旧 `/api/v1/sh000001/latest.csv` 地址也返回相同 JSON。归档完整性校验失败时返回 `503`；新发布仍须通过历史账本的一致性检查。

面板分为预测概览、历史明细、运行任务、快照归档和研究对照。概览与明细支持近 5、20、60 个交易日及自定义 1 至 5000 日统计，选择保存在登录会话中。统计读取当前模型账本内截至该发布日的已结算历史，不受 CSV 的 60 日导出窗口限制；样本不足时显示实际可用数量。待结算预测不计入统计。平衡准确率按实际上涨/下跌的召回率平均计算，只有单一实际方向时显示为空。

控制面板保留自定义统计天数；公开统计 API 的 `metrics` 和 `recent_20` 始终使用当前正式模型。管理员模型对比页另行显示四模型在相同目标交易日上的统计。组合由 `[服务] 模型组合目录` 指定，不能仅通过修改底层基线引擎字段来切换正式模型。

2026-09-19 已按用户要求撤回强制连续看涨保护：删除“第九次反向”、滚动看涨次数配额、固定 -0.01% 幅度和 2025 年生效日期等覆盖输出的规则，并移除其发布入口。正式模型恢复到 `956dd395b7f9df52a46bf6f739f81eb5a1b738c1180d72abfc7c741bdf6522e9`。原 `bullish_streak_guard_v1` 至 `v4` 的报告保留为撤回记录；其中的“26/26”不构成有效的独立性能验证，此前“看涨偏向已解决”的结论已撤回。详见 [撤回说明](artifacts/evaluation/bullish_guard_withdrawal_20260919/README.md)。

后续还验证了候选组合、滞后一日的融资与估值特征、2011 年起的长历史训练及既有替代引擎，结果同样未达到验收要求。新增行情和逐日结果仅保存在 `artifacts/evaluation`，没有覆盖生产输入或已发布记录。

新发布的 CSV 在原列之后增加 `预测目标交易日`、`记录来源` 和 `预测生成时间`。`live` 表示在目标交易日开盘前生成，`backfill` 表示事后回放，日历不足时标记 `unknown`。日期依据持久化的 SSE 日历；无法确认时不会用普通工作日猜测。历史归档保持原字节内容，旧格式浮点读取保持兼容，新快照使用精确浮点往返解析。

管理员点击“刷新数据与预测”后，后台任务会从 Tushare 拉取数据、拒绝覆盖既有历史输入、补齐上一条预测的实际结果、生成新预测，并把特征、CSV、原始行情和哈希清单归档到 `service_data/archives`。填写 `[Tushare]` 的 `令牌` 后，将 `[服务]` 的 `启用定时刷新` 设为“是”，即可在上海时间 `刷新小时:刷新分钟` 自动提交每日刷新。

定时任务根据 SSE 日历判断应有数据截止日，并补跑错过的交易日。行情未增长会记录为 `skipped`；失败或行情晚到时，间隔至少 10 分钟重试，每个目标交易日最多 6 次。服务对数据目录持有独占锁，只允许一个实例刷新同一本账本；启动时将遗留任务标记为 `interrupted`，刷新任务每 10 秒记录心跳。`/healthz` 同时检查归档可读性与数据新鲜度，数据过期、缺失或日历未知时返回 `503`；`/livez` 只表示进程存活。State Veto 归因位于研究对照页，只计算已结算记录。

旧 BiLSTM 影子入口保留为独立研究工具，默认关闭；新的三个影子模型随正式组合自动运行，无需启用 BiLSTM。BiLSTM 研究结果写入独立版本和归档，不参与正式组合的计算。

在 `config.ini` 的 `[BiLSTM影子]` 中将 `启用` 改为“是”，并按需调整 `循环验证天数` 和 `重训间隔交易日` 后重启服务。

管理员面板可手动提交一次影子任务、查看其状态、最近 60 日胜率与平衡准确率，并下载影子 CSV。影子任务失败只记录失败原因，不会影响已发布的默认预测。

临时使用其他端口可运行 `python -m scripts.run_service --port 8010`，`--no-scheduler` 可仅在本次启动关闭定时任务。浏览器回归工具 `python tools/verify_service_panel.py --url http://127.0.0.1:8010` 需要另行安装 Playwright 及 Chromium，验证统计窗口、筛选、分页、CSV 缓存与桌面/手机布局，截图写入 `artifacts/run/service-panel/`。工具从本地配置读取登录凭据，不输出密码。

对外部署时，在 `config.ini` 中将 `[服务]` 的 `监听地址` 设为 `0.0.0.0`，并通过 HTTPS 反向代理暴露服务；反向代理正确终止 TLS 后，再将 `Cookie仅HTTPS` 设为“是”。生产环境应将 `数据目录` 指向受备份保护的持久化磁盘，并保留其 `archives` 子目录的全部版本。自定义配置路径可使用 `python -m scripts.run_service --config D:\secure\prediction.ini`。

入口按职责放在 `scripts/` 下：`predict.py` 负责次日预测，`loop_validation.py` 负责历史回放，`shadow_predict.py` 和 `shadow_validation.py` 负责固定因果 BiLSTM 影子模式，`fetch_akshare.py`、`fetch_tushare.py` 和 `fetch_wind.py` 负责数据接入，`run_service.py` 负责启动服务。模块形式调用（`python -m scripts.<name>`）可确保项目根目录被正确加入导入路径。

预测引擎源码或固定参数发生变更后，服务不会覆盖旧预测。刷新时会先对活动归档重算，只有与旧账本逐字段一致时才执行兼容发布升级。改变模型预测方向的研究不能借此发布。命令行升级需先停止服务以释放数据目录独占锁：

```powershell
python -m scripts.run_service --promote-compatible-release
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

## 固定看涨纠偏候选

`tools/run_downside_shadow.py` 冻结并运行一个已经固定参数的资金流向与价格组合候选。它使用独立的模型 release、预测账本和 `service_data/shadow_archives`，不会修改公开 CSV、活动 publication 或正式模型。冻结时会校验原研究结果仍为 25/26 项通过，不能把失败候选伪装成已验收模型：

```powershell
python tools/run_downside_shadow.py freeze --bundle artifacts/run/new_moneyflow_price_shadow
python tools/run_downside_shadow.py run --bundle artifacts/run/new_moneyflow_price_shadow --fetch
```

运行必须在信号日上海时间 18:00 之后、下一交易日开盘之前完成，才能记为事前预测。已经开盘的目标日默认拒绝；显式 `--allow-backfill` 才允许补录，补录行排除在事前配对统计之外。使用 `--fetch` 时，增量采集器按缺失交易日分页获取个股数据，以沪深 A 股计算特征（排除北交所代码）；新响应保存收取时间和哈希。原始资金流向和包含价格组合的两套模型都训练完成后，才允许改判。

命令基于服务已经发布的最新行情快照运行，不主动刷新正式行情或启动定时任务。后续每个交易日正式行情发布后运行同一 bundle；其源码、参数或依赖改变会被拒绝，需要显式冻结新版本。控制面板不增加此候选的展示或操作。该候选仍为独立研究版本，没有生产晋升入口。

当前已冻结并联调的 bundle 为 `artifacts/evaluation/moneyflow_price_shadow_v1`。其原验收为 25/26 项通过，两个独立历史前缀逐字段复现；首次真实运行发生在目标日开盘后，明确标记为回放，事前配对样本数为零。详细证据见 [docs/DIRECTION_BIAS_REVIEW.md](docs/DIRECTION_BIAS_REVIEW.md)。

2026-09-19 05:15（上海）已记录该固定候选对 2026-09-21 的一条真实事前预测，仍未结算，事前配对已结算样本数仍为零。它没有通过原有全部历史门槛，也未替换生产。当前日内数据研究缺少有权限的 `idx_mins` 数据入口；本地构建器已准备好，但软件测试不构成模型成绩。最新证据见 [direction_bias_status_20260919.json](artifacts/evaluation/direction_bias_status_20260919.json)。纠偏目标尚未完成。

可使用独立后台记录器，在正式快照更新后自动运行同一固定候选：

```powershell
python tools/watch_downside_shadow.py run --bundle artifacts/evaluation/moneyflow_price_shadow_v1
python tools/watch_downside_shadow.py status
python tools/watch_downside_shadow.py stop
```

默认每 60 秒检查一次；当前快照已有记录时不重复取数或写预测。临时失败每十分钟重试，同一快照最多六次，重启后保留次数。错过开盘窗口就跳过，不自动补录。`status` 检查实际进程锁及心跳年龄；`stop` 等当前计算和账本写入结束后停止。记录器不负责更新正式行情，仍依赖现有服务的刷新任务及机器持续运行；未注册开机自启。状态文件为 `service_data/downside_watcher.json`，不在控制面板增加功能，也不自动替换生产模型。


## 开发与部署检查

软件回归和需要独立研究归档的验收分别运行，见 [docs/TESTING.md](docs/TESTING.md)。测试依赖在 `requirements-test.txt` 中声明；冻结运行包受 `.gitattributes` 保护，必须保留原始字节和清单哈希。

Linux 部署文件位于 `deploy/`。`predict-index.service` 使用私有文件权限掩码；配置加载时也会将数据目录限制为 `0700`、持久化会话密钥限制为 `0600`。证书续期使用现有 Nginx ACME webroot 路由，仅在实际续期成功且 Nginx 配置检查通过后 reload。
