# 正式模型与影子模型

用户于 2026-09-19 明确选择将期权＋资金流组合用于生产。此为用户指定模型的正式发布，不代表原历史研究验收全部通过；三个候选的原报告保持不变，强制翻转连续看涨方向的旧规则不参与计算。

| 标识 | 名称 | 角色 | 固定实验参数 |
| --- | --- | --- | --- |
| option_moneyflow | 期权＋资金流组合 | 正式生产 | Logistic；55% 阈值；资金流与期权概率等权；共同可用样本；当日期权活动和持仓特征 |
| baseline | 原生产基线 | 影子 | 原 state_veto_rule 算法和参数 |
| moneyflow | 资金流信号 | 影子 | Logistic；60% 阈值；前一交易日资金流 |
| moneyflow_price | 资金流＋价格特征组合 | 影子 | Logistic；55% 阈值；资金流与价格扩展模型概率等权 |

三个候选保留原有 756 日训练窗口、120 条最少训练样本、5 个交易日重训间隔、因果置信度和收益幅度校准。运行时不重新选择候选、阈值、训练年份或连续方向限制。底层依赖版本与科学计算源码在模型包中固定，参数或代码变化必须生成新版本。

当前冻结环境为 Python 3.12、NumPy 2.4.6、pandas 2.3.1、scikit-learn 1.7.1、SciPy 1.17.1、XGBoost 2.1.2、PyTorch 2.5.1+cu121。`requirements-service.txt` 声明服务依赖范围；部署既有冻结包时还须匹配其 `manifest.json` 中的实际依赖版本，运行时会校验，不能直接升级依赖后继续沿用同一模型版本。

## 每日流程

共享行情 → 增量补齐上下文 → 计算一次完整基线 → 计算四个固定预测序列 → 核对各自既有账本 → 归档 → 一次事务写入四个模型并更新正式发布指针。

每天上海时间 **20:15** 调用同一刷新入口。此时间来自当日期权模型最早 20:00 发布的原实验约定。行情、市场宽度和期权活动使用当日已完成数据；资金流使用前一国内交易日；美股收盘使用早于国内信号日的日期。期权持仓变化使用相邻交易日的相同合约。抓取只补齐缺失日期，并保存原始响应、接收时间与哈希；市场宽度与资金流会按缺失日期获取全市场日线截面，不重新下载全部个股历史。

四模型作为一个发布批次：任何计算、历史一致性或归档校验失败，本轮不会覆盖公开结果，刷新任务显示失败并按现有调度策略重试。公开接口只读取已发布文件，不触发训练或抓取。旧预测和实际结算继续保持不可变；新运行包使用新版本账本，不给新增模型继承其他模型的预测时间。

## 目录职责

- `prediction_service/forecast_models.py`：四个固定算法的科学计算适配。
- `prediction_service/model_registry.py`：固定模型定义、来源、版本指纹和逐字段一致性验证。
- `prediction_service/model_context.py`：共享信号增量获取。
- `prediction_service/portfolio.py`：批次发布、账本持久化和离线重算。
- `prediction_service/model_comparison.py`：同日期统计、事前样本筛选和管理员导出。
- `artifacts/releases/four_model_portfolio_20260919/`：冻结源码、依赖、种子数据、原实验报告与逐字段一致性证据。
- `service_data/archives/`：每轮共享输入、四模型完整回放和各 61 行 CSV。

旧独立 `moneyflow_price_shadow_v1` 模型包和记录保留为历史研究证据，新的每日任务由统一组合运行。不要并行启动旧影子观察脚本。

## 管理员使用

`/admin/models` 显示四模型下一交易日的预测、同区间准确率、平衡准确率、上涨与下跌召回率、MAE、RMSE、累计准确率曲线及逐日对照。可选择 20/60/252 日或输入 1～5000 个交易日；可仅显示方向分歧。

统计窗口以生产模型最近的目标交易日为准，只比较四个模型均有记录的日期。缺失日期单独提示，不使用更早日期补足。`仅事前预测` 要求四个模型的记录均在目标日开盘前生成；历史回放不计入，样本为零时显示等待结算，不用回测成绩填充。

- 公开 CSV：`GET /api/v1/sh000001/latest.csv`，60 个已结算交易日＋下一交易日，原列结构与缓存协议保留。
- 管理员对比数据：`GET /admin/api/models?days=60&sample=all`。
- 管理员对比 CSV：`GET /admin/models/comparison.csv?days=60&sample=all`。
- 管理员各模型 CSV：`GET /admin/models/{model_key}/latest.csv`。

后三个接口均需要管理员会话。`sample=live` 仅比较事前记录。逐日对比导出包含预测方向、涨跌幅、点位、置信度、命中、来源和生成时间。

## 维护与发布

以下命令必须在项目根目录运行。准备阶段不改变生产状态；已有模型目录拒绝覆盖。

```powershell
python tools/manage_model_portfolio.py prepare --bundle artifacts/releases/four_model_portfolio_20260919
```

激活前先备份 SQLite、配置和现有发布标识，并停止使用同一数据目录的服务进程。激活命令持有服务文件锁；不同算法的切换不能使用旧的“兼容发布”按钮。

```powershell
python tools/manage_model_portfolio.py activate --bundle artifacts/releases/four_model_portfolio_20260919 --fetch
```

激活成功后配置 `[服务] 模型组合目录 = artifacts/releases/four_model_portfolio_20260919`，并设定刷新时间 20:15、启用定时刷新，再启动 `python -m scripts.run_service`。不应只修改底层 `正式预测引擎` 字段。

离线重算不访问 Tushare，核对全部模型账本及归档 CSV：

```powershell
python tools/manage_model_portfolio.py verify --bundle artifacts/releases/four_model_portfolio_20260919
```

`--data-dir` 可以使用已通过 SQLite backup 复制的隔离数据库进行发布演练。模型切换不是删除历史记录；如需恢复此前生产版本，应使用保留的配置、发布快照及完整数据库备份，在服务停止时操作，不修改预测账本。
