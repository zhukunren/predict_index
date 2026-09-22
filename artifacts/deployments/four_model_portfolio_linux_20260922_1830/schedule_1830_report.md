18:30 更新时间切换结果 - 2026-09-22（上海时间）

数据拉取和四模型预测更新时间已切换为上海时间 18:30，生产服务已重启并运行新冻结包。当前公网 /healthz 返回 status=ok、runtime_matches=true，不再返回 model_mismatch。

模型运行包为 artifacts/releases/four_model_portfolio_linux_20260922_1830，Bundle ID 为 14044eac4323f3f5d953796c503e3f3d70d19df9f4e947f010f08d3fb71f4250；正式算法仍为 fixed_option_moneyflow_v1。本次只是发布时点契约升级，独立复算确认四个模型各 1,400 行预测的日期、方向、收益、价格、缩放系数和概率结果均保持一致。旧运行包未修改，旧账本和旧归档保留。

当前生产标识：

- 快照：71a78f92-7d9c-4c9b-854c-90e642c592bd
- 正式 Release：72bb51a239bf19caddc432ce3440a485da4c8ae454cd087b7ba3d99ae9506114
- CSV SHA-256：0597b27ebb0a4aedde25faf978cbab70055dad174ba17273e80d2a17148e36a0
- 数据截止日／应有截止日：20260921／20260921
- 定时刷新：18:30（上海时间）

服务当前为 active/running，PID 1215505，自动重启次数 0，UMask=0077。配置权限为 0600，数据目录为 0700，会话密钥为 0600。切换前备份位于 service_data/backups/before_schedule_change_20260922T1720，未打印或复制凭据内容。

完整软件回归：459 passed，2 deselected，35 warnings。公网健康接口已核对两次，均返回新快照和 runtime_matches=true。本次未等待 18:30 的下一轮真实拉取，因此当前数据仍截止 20260921；下一交易日刷新将使用新的 18:30 调度时间。

证据：

- [18:30 运行包独立复算](/data/projects/predict_index/artifacts/run/audit_fixes_20260922/schedule_1830_bundle.json)
- [18:30 生产验收](/data/projects/predict_index/artifacts/deployments/four_model_portfolio_linux_20260922_1830/schedule_1830_production.json)
- [完整软件回归 JUnit](/data/projects/predict_index/artifacts/run/audit_fixes_20260922/tests.xml)
