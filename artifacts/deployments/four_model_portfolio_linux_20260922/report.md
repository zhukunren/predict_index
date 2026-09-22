**项目与服务修复结果 — 2026-09-22（上海时间）**

代码与服务修复已部署至 `https://predict.dwzq.top`，17:00 最后核对时健康检查返回 `200`、`runtime_matches=true`。正式模型为“期权＋资金流”，基线、资金流、资金流＋价格为三个影子模型。软件回归通过；两项需要独立原始研究归档的验收尚未执行。

| 审查问题 | 修复及验证 |
| --- | --- |
| 连续刷新误报历史修订 | 仅允许八种指定的可选恒生滞后特征补齐已知缺口，旧版本仍保留冻结缺失值；历史非空值与其他字段修订继续拒绝。修复混合缺失值的数值比较。连续追加、相同输入重试、真实归档行情重建均通过。 |
| 原冻结包字节不匹配 | 恢复 18 个 seed 文件原始 CRLF 字节，全部匹配原 manifest；没有修改原清单哈希。新增 Git 字节保留规则，原包和新包完整性校验均通过。 |
| 实际生产与四模型说明不符 | 使用独立 Linux 运行包完成隔离演练、生产显式激活和四模型严格复算；健康接口增加发布版本、算法与运行配置一致性检查。 |
| 会话签名密钥权限过宽 | 数据目录 `0700`、密钥 `0600`、服务 `UMask=0077`，加入文件锁与防符号链接保护。生产密钥已轮换，旧 Cookie 被拒绝，新登录通过。 |
| 证书检查停止 Nginx | 改用现有 HTTP webroot，只有实际续期成功且 `nginx -t` 通过才 reload。真实检查执行成功、公网 ACME 路由通过；证书剩余 89 天，本次未触发新证书签发。Nginx PID 保持 `1026224`。 |
| 测试依赖及研究证据混用 | 声明测试依赖，默认运行软件测试；两项原研究验收独立标记，缺文件时显式验收仍失败。新增合成 fixture 仅验证加载器行为，不代替研究证据。 |

Linux 迁移使用新版本账本，原包与旧账本继续保留。迁移核对原 1,397 个日期，方向、收益、价格和缩放系数完全一致；仅概率允许绝对误差不超过 `1e-12`，实际最大 `1.1590728377086634e-13`。新包另行独立重算并严格逐字段匹配，未放宽原包校验或改变历史研究门槛。

当前四个新模型各有 1,400 条预测。旧基线的 63 条预测、62 条结算及旧归档字节均保持不变。四模型新写入的历史记录如实标记为回放；共同事前样本为 `0`，不能将历史回放成绩作为事前成绩。

| 生产标识 | 值 |
| --- | --- |
| 模型包 | `artifacts/releases/four_model_portfolio_linux_20260922` |
| Bundle ID | `6aad1b80bc4123c6fdd1b748e62515fb7b89151c079d83bdb45c25ca65f13b42` |
| 正式算法 | `fixed_option_moneyflow_v1` |
| 正式 Release ID | `9927f8d45a3913eb4feb6e3af6a61681759a6f20baf9f87aa52abf84c5195abe` |
| 当前快照 | `37c21278-3f2b-4751-9765-eae70c58bd0d` |
| CSV SHA-256 | `8c4e12937302def69e7dea91e04900906a736b2afd62100e4ba3728971a57546` |
| 数据截止日／应有截止日 | `20260921`／`20260921` |
| 定时刷新 | 已启用，上海时间 `20:15` |

软件回归 **459 passed，2 deselected，35 warnings**，耗时 153.77 秒；`pip check` 与 `git diff --check` 通过。生产服务 16:51:40 启动后自动重启次数为 0，日志抽查无新错误。公开 CSV 共 61 行，与管理员正式模型下载一致；ETag 内容哈希、HEAD、条件请求 `304` 均通过。四模型管理员接口及三个管理页面验收通过。

两项未执行的独立研究验收需要恢复 `artifacts/evaluation/bilstm_causal_v1/` 和 `artifacts/evaluation/ensemble_moneyflow_price_downside_v1/` 的原始证据，具体文件与执行命令见 [测试说明](/data/projects/predict_index/docs/TESTING.md)。今日 20:15 的自动刷新尚未发生，本次验证使用截至 9 月 21 日的归档数据和独立重算。

管理员需重新登录。修复前完整私密备份位于 `service_data/backups/before_audit_fixes_20260922T160754/`；激活前停服务备份位于 `service_data/backups/before_portfolio_activation_20260922T164824/`。如需回滚，应停止服务后恢复匹配的代码、配置、数据库和归档；已轮换的旧会话密钥仅留作受限审计，不应重新启用。

验收证据：

- [生产激活与旧记录保留核对](/data/projects/predict_index/artifacts/deployments/four_model_portfolio_linux_20260922/production_activation.json)
- [公网 HTTP 验收](/data/projects/predict_index/artifacts/deployments/four_model_portfolio_linux_20260922/http_acceptance.json)
- [服务、权限、模型包及健康检查](/data/projects/predict_index/artifacts/deployments/four_model_portfolio_linux_20260922/final_checks.json)
- [跨平台数值比较](/data/projects/predict_index/artifacts/deployments/four_model_portfolio_linux_20260922/linux_seed_parity.json)
- [软件回归 JUnit 结果](/data/projects/predict_index/artifacts/deployments/four_model_portfolio_linux_20260922/tests.xml)

工作区原有改动保留，本次未提交或推送 Git。
