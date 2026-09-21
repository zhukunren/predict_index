# 连续看涨核查：2026-09-17

**2026-09-19 后续发布：用户明确指定期权＋资金流组合作为正式模型。** 原生产基线、资金流信号、资金流＋价格组合已接入统一影子运行。此次发布保留原实验参数及未完全通过的研究验收结论，不表示已经证明“偏向解决且性能不回退”。当前架构见 [模型组合说明](MODEL_PORTFOLIO.md)，生产标识和验证证据见 [发布审计](../artifacts/deployments/four_model_portfolio_20260919/deployment_audit.json)。下文及早先状态文件对“正式模型”的描述属于切换前的历史核查记录。

**2026-09-19 更正：强制方向保护已撤回，原任务尚未完成。** 此前新增的连续次数上限、看涨配额、固定下跌幅度及按开发/回归日期启停的规则不满足用户要求。`bullish_streak_guard_v1` 至 `v4` 的通过数字不能证明模型识别能力改善；相关成功结论全部撤回。代码和正式发布恢复原算法，最新行情保留。恢复证据与完整原因见 [撤回说明](../artifacts/evaluation/bullish_guard_withdrawal_20260919/README.md)。

正式模型确实存在连续 15 次看涨。问题出在规则对近期市场状态的判断持续同向，当前证据没有显示 CSV 缓存、日期错位或网页重复渲染导致该现象。连续看涨本身不能证明模型错误，但近期下跌识别能力不足，单独展示总体准确率容易掩盖这一点。

## 正式发布的结果

数据来源：快照 `712b07e3-bcfc-4378-9b50-27ea68ed08cf` 的 `results.csv` 和 `diagnostics.csv`，正式算法 `rank_weighted_state_veto_v1`，行情截至 2026-09-17。以下仅统计已结算预测。

| 范围 | 预测上涨 | 实际上涨 | 方向准确率 | 始终看涨基准 |
| --- | ---: | ---: | ---: | ---: |
| 近 20 个交易日 | 18/20，90.00% | 10/20，50.00% | 60.00% | 50.00% |
| 近 60 个交易日 | 32/60，53.33% | 33/60，55.00% | 58.33% | 55.00% |
| 连续看涨区间 | 15/15，100.00% | 10/15，66.67% | 66.67% | 66.67% |

连续区间的信号日是 2026-08-19 至 2026-09-08，目标交易日是 2026-08-20 至 2026-09-09，共 15 次预测。区间内有 5 次实际下跌，均未识别。近 20 日实际下跌 10 次，只识别 2 次，下跌召回率为 20.00%；近 60 日的整体分布会掩盖这种近期集中。

近 60 日只有 35 次命中，始终看涨也能命中 33 次，差距为 2 次、3.33 个百分点。这不足以证明有稳定的预测优势。当前这 60 条是历史回放记录，不是提前逐日存证的实盘成绩；待结算的 2026-09-18 预测不参与这些指标。

## 规则层原因

`循环验证脚本.py` 中 `_nested_volatility_rule_signal` 按最近 120 个已知结果的准确率排序，取前 5 个规则名加权投票。互补规则经取反后可能成为相同路径，目前将它们合并显示，但仍累加原有排名权重；因此这段时间的 5 个名额实际只有 3 条不同路径，不代表 5 份独立证据。

归档诊断逐日显示：

- 15 天的 `base_predicted_label` 都为 1，最终标签也均为 1。
- 收盘位置规则入选 15 天，20 日波动率规则入选 11 天，布林带宽度规则入选 8 天。相关或较缓慢变化的特征反复主导投票。
- `veto_applied` 和 `recent_failure_guard_applied` 都是 0。
- 失败保护观察到的近 15 日原始准确率为 53.33% 至 73.33%，没有达到 45% 的降级条件，更未达到反转条件。该保护监测总体失效，未检查单方向集中或下跌召回率。

方向取决于规则投票，幅度校准与置信度校准不负责打断同向序列。这里的结果符合现有规则；需要研究的是规则的区分能力、评分基准和相关专家权重，不能通过强制每隔几天反向来修饰结果。

## 影子模型不构成直接替代方案

最新已完成的 BiLSTM 影子归档截至 2026-09-15，与正式最新窗口并不完全相同。它的近 60 条已结算预测中 50 次看涨、实际上涨 34 次，准确率 50.00%，低于该窗口始终看涨的 56.67%；最长连续看涨为 30 次。

已有冻结对照 `artifacts/evaluation/bilstm_causal_v3` 的 412 个信号日中，正式模型和 BiLSTM 准确率同为 58.01%，但 BiLSTM 方向切换率只有 8.27%，正式模型为 41.85%。该对照中 BiLSTM 最长连续看涨达 54 次，因此更换为现有影子模型并不能解决方向黏滞问题。冻结对照和最新影子归档的输入及运行记录应分别解释。

## 模型纠偏验收

按用户要求，已经撤下新增的面板方向诊断和集中提示，保留自定义统计天数。以下工作针对预测算法；截至本次验收，尚未找到同时减轻近期看涨偏向且性能不回退的候选，未改变正式模型或已发布 CSV。

第一轮在开发期（2023-01-03 至 2024-12-31，484 个信号日）比较三个预先冻结的候选：

| 模型 | 开发期准确率 | 开发期平衡准确率 |
| --- | ---: | ---: |
| 原模型 | 51.45% | 51.47% |
| 按原准确率评分，去重后取 5 个专家且不重复加权 | 50.00% | 50.00% |
| 按平衡准确率评分，并去重 | 48.35% | 48.36% |
| 按平衡准确率评分，每个特征只占一个名额 | 46.28% | 46.28% |

按开发期规则选出的去重候选继续接受同日对照验收。后续回归期为 2025-01-02 至 2026-09-16 的 415 个信号日，最后结果由 09-17 收盘价结算。另检查最近 252、60、20 日；这些近期窗口与回归期重叠，不是独立证据。

| 范围 | 原模型准确率 | 去重候选准确率 |
| --- | ---: | ---: |
| 415 日回归期 | 58.07%（241/415） | 53.25%（221/415） |
| 近 252 日 | 58.73% | 53.17% |
| 近 60 日 | 58.33%（35/60） | 51.67%（31/60） |
| 近 20 日 | 60.00%（12/20） | 65.00%（13/20） |

该候选近 60 日最长连涨从 15 次降至 11 次，但回归期修正 38 次原有错误，同时新增 58 次错误；平衡准确率、幅度误差和置信度 Brier 也未通过不回退检查。近期好转不满足整体不回退要求。冻结合同、源代码、输入和逐日结果位于 `artifacts/evaluation/direction_debias_v1/`，结论为 `report.json` 中的 `passed: false`。

第二轮为单一、固定参数的历史分歧纠偏层：对每个候选，仅考察此前 252 日内“原模型与候选意见相反、且原模型方向与当前相同”的已结算记录；至少 20 个分歧样本、Wilson 95% 区间下界大于 50%，并且近 60 日至少 5 个同类分歧、候选命中率不低于 50%，才允许纠正原方向。涨跌对称处理，不强制交替。纠偏后的幅度和置信度使用自身过去的混合预测重新校准。

纠偏层在 2023-2024 开发期改判 17 次，开发期准确率从 51.45% 降至 50.00%。在 2025 年起的回归期没有任何日期满足纠偏证据要求，方向与原模型完全相同：24 项性能条件通过，但看涨占比差和最长连涨两项纠偏条件失败。它同样不能解决当前问题。有效结果位于 `artifacts/evaluation/direction_guard_v2/`；`direction_guard_v1/` 因实现阶段发现数组别名问题而作废，禁止用作验收依据，未参与参数选择。

两轮验收对四个窗口同时要求准确率、平衡准确率、下跌召回率不下降，以及涨跌幅 MAE、RMSE 和置信度 Brier 不升高；还要求近 20 日预测与实际上涨占比差缩小、近 60 日最长连涨缩短。机器报告逐项保存检查结果。没有通过全部条件的模型被发布。

这些历史数据已在既有项目研究及本次分析中被查看，因此称为回归验收，不将其包装成全新的未见测试集，也不据此保证未来性能。本次没有在读到后续比较结果后调参重试三种规则候选或纠偏层。

复现命令（输出目录必须不存在）：

```powershell
python tools/evaluate_direction_bias.py --input artifacts/evaluation/direction_debias_v1/features.csv --output artifacts/run/new_direction_bias
python tools/evaluate_direction_guard.py --input artifacts/evaluation/direction_guard_v2/features.csv --output artifacts/run/new_direction_guard
```

未通过验收时工具返回非零状态，且始终不会修改服务配置或发布数据。候选、纠偏层及门槛的测试覆盖重复权重、当前/未来结果隔离、前缀回放一致性、原模型历史不可变、日期和标签对齐，以及不纠偏时不得误报成功。

## 扩展实验结果

后续增加了概率投票、跨指数特征、误差预测和少量高置信度改判。每一轮仍先冻结参数及输入，再只用 2023、2024 年选择候选；2025 年以后的数据只作已观察过的回归检查。全部实验保存在独立目录，没有覆盖前一轮结果，也没有降低上述 26 项验收条件。

| 实验目录 | 方法与选中候选 | 415 日准确率 | 近 60 日准确率 | 近 60 日最长看涨 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 原模型 | `rank_weighted_state_veto_v1` | 58.07% | 58.33% | 15 | 对照 |
| `direction_probability_v1` | 保留原排名，改为规则条件概率；开发期选中逻辑回归聚合 | 54.70% | 53.33% | 15 | 不通过 |
| `context_residual_v1` | 沪深 300、中证 500、创业板特征及 XGBoost 残差；选中平衡权重版本 | 53.25% | 51.67% | 9 | 连续看涨缩短，但性能回退 |
| `context_error_v1` | 直接学习原预测是否出错；选中 756 日训练版本 | 53.25% | 58.33% | 15 | 不通过 |
| `context_selective_v1` | 对以上六个模型分别试验预先固定的五个改判置信度门槛 | 未运行 | 未运行 | 未运行 | 30 个组合均未通过两个开发年份的筛选 |
| `directional_error_v1` | 小型逻辑回归，加入方向各自的历史误差；选中 756 日训练版本 | 58.55% | 61.67% | 15 | 准确率提高，但没有纠偏 |

跨指数输入来自同一交易日收盘行情。`direction_context_20260917/manifest.json` 记录三个指数各 1,628 行的日期覆盖和文件哈希；它们没有被写入生产行情目录。树模型使用原预测的因果校准概率作为先验，只从此前已结算的标签学习，每五个交易日重训。

高置信度筛选采用 0.55、0.60、0.65、0.70、0.75 五个门槛，要求两个开发年份的准确率和平衡准确率分别不下降，且开发期至少改判五次。没有组合合格，所以该轮止于开发期，没有再按近期表现挑选门槛。

最后的小型误差模型使用十个输入，包括原方向和置信度、此前市场收益，以及过去 120 日看涨、看跌预测各自的命中率。方向命中率用十个伪样本向 50% 收缩，所有统计均排除当前结果。两个预先固定的训练窗口为 252、756 日；只有反向概率达到 60% 才改判。标准化仅拟合此前训练样本。

756 日版本通过两个开发年份的方向筛选，回归期修正五次错误、新增三次错误。虽然整体准确率和平均误差改善，但近 60 日下跌召回率从 55.56% 降至 48.15%，看涨次数从 32 次增加至 38 次，最近 20 日仍为 18 次看涨，最长连续看涨仍为 15 次。近 20 日幅度 MAE、RMSE 也上升，合计七项验收失败。因此不能把该候选称为看涨纠偏成功，也未将其发布。

新测试验证真实 XGBoost 和逻辑回归训练的当前/未来标签隔离、完整历史与前缀回放一致性、上下两个方向对称改判、跨指数缺失日期拒绝，以及输入预测保持不变。原始预测源码与服务面板没有因这些实验而改动。

新增复现入口（输出目录必须不存在）：

```powershell
python tools/evaluate_direction_bias.py --family probability --input artifacts/evaluation/direction_debias_v1/features.csv --output artifacts/run/new_direction_probability
python tools/evaluate_context_residual.py --input artifacts/evaluation/direction_debias_v1/features.csv --context artifacts/evaluation/direction_context_20260917 --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_context_residual
python tools/evaluate_context_residual.py --family correctness --input artifacts/evaluation/direction_debias_v1/features.csv --context artifacts/evaluation/direction_context_20260917 --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_context_error
python tools/evaluate_selective_context.py --direction-run artifacts/evaluation/context_residual_v1 --correctness-run artifacts/evaluation/context_error_v1 --output artifacts/run/new_context_selective
python tools/evaluate_directional_error.py --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_directional_error
```

截至这些实验结束，尚无候选同时满足纠偏和性能不回退。继续反复使用同一已观察回归区间挑选模型会增加过拟合风险；即使后续有候选通过，也仍需保持独立版本，并积累真实事前预测的结算记录。

## 新增数据与既有引擎

随后针对三个不同假设进行了独立实验：候选之间的分歧是否提供额外信息、资金与估值数据是否有助于识别下跌、更多早期市场周期是否改善学习。仍先冻结每轮参数和输入，要求 2023、2024 年分别不回退，未通过的候选不进入后续回归比较。

| 实验目录 | 候选 | 2023 年准确率 | 2024 年准确率 | 开发期筛选 |
| --- | --- | ---: | ---: | --- |
| 固定对照 | 原模型 | 48.35% | 54.55% | 对照 |
| `direction_stack_v1` | 八个因果预测源的组合误差模型 | 51.24% | 52.48% | 不通过 |
| `direction_stack_v1` | 组合误差模型，涨跌平衡权重 | 50.83% | 52.48% | 不通过 |
| `direction_funding_v1` | 资金与估值方向模型 | 51.65% | 52.89% | 不通过 |
| `direction_funding_v1` | 资金与估值方向模型，平衡权重 | 50.83% | 53.72% | 不通过 |
| `direction_funding_v1` | 资金与估值误差模型 | 41.32% | 52.07% | 不通过 |
| `history_direction_v1` | 长历史方向模型 | 50.83% | 51.65% | 不通过 |
| `history_direction_v1` | 长历史方向模型，平衡权重 | 50.00% | 51.24% | 不通过 |
| `history_direction_v1` | 长历史误差模型 | 46.69% | 53.72% | 不通过 |
| `core_direction_engines_v1` | 既有 `stability_rule` | 52.89% | 50.00% | 不通过 |
| `core_direction_engines_v1` | 既有 `nested_ml` | 50.00% | 56.20% | 通过开发期，后续回归失败 |

组合模型使用全部六个跨指数模型和两个小型误差模型的历史预测概率，只在辅助模型均已训练后开始学习。辅助模型的训练日期必须早于各自信号日，组合模型也只使用此前已结算的结果。标准化和涨跌权重仅从当时的训练区间拟合。

资金数据冻结在 `direction_funding_20260917/`：指数换手率与估值覆盖 1,872 日，上海市场融资融券覆盖 1,871 日。最新融资记录为 2026-09-16。所有新增特征统一滞后一个国内交易日，缺失中间日期会直接拒绝计算，不能用更早的记录静默填补。`funding_selective_v1/` 进一步沿用此前固定的五个置信度门槛，三个资金模型的 15 个组合均未通过两个开发年份的筛选。

长历史数据冻结在 `direction_history_2011/`，从 2011-01-04 至 2026-09-17 共 3,817 个交易日，原 2020 年起的 1,628 行输入保留不变。上证、恒生及三个国内指数的原始获取记录和哈希都在目录内。共同样本从 2011 年起算，避免使用创业板指数推出前并不存在的数据。`direction_history_2010/` 和 `direction_history_2010_v2/` 是日期格式及指数可用区间校验中止的拉取尝试，不是有效模型验收。

长历史回放只补齐原冻结预测以前的 2,189 个信号，原有 1,397 个预测逐字段保留，新增与既有结果的收益标签均完成精确对齐，见 `history_direction_v1/alignment.json`。三个长历史候选将训练窗口扩展为 2,520 日，其余 XGBoost 设置沿用此前固定设置，但仍未通过开发期筛选。

既有引擎对照没有调参。`nested_ml` 在开发期分别提高两年的准确率和平衡准确率，因此继续接受同日回归验收。其 415 日准确率为 49.16%，低于原模型 58.07%；近 60 日为 55.00%，低于 58.33%；近 20 日为 50.00%，低于 60.00%。近 20 日看涨数量从 18 次降到 10 次，说明可以改变方向分布，但该改进伴随预测性能回退，不能发布。

新增验证覆盖组合模型的前缀回放与未来标签隔离、融资数据的发布滞后和缺失日期、平衡权重树模型的因果性。上述实验不改变正式预测源码、公开 CSV 或面板内容。

复现新增验收（输出目录必须不存在）：

```powershell
python tools/evaluate_stacked_direction.py --direction-run artifacts/evaluation/context_residual_v1 --correctness-run artifacts/evaluation/context_error_v1 --linear-run artifacts/evaluation/directional_error_v1 --output artifacts/run/new_direction_stack
python tools/evaluate_context_residual.py --family funding --input artifacts/evaluation/direction_debias_v1/features.csv --context artifacts/evaluation/direction_context_20260917 --funding artifacts/evaluation/direction_funding_20260917 --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_direction_funding
python tools/evaluate_selective_context.py --family funding --direction-run artifacts/evaluation/direction_funding_v1 --output artifacts/run/new_funding_selective
python tools/evaluate_history_direction.py --history artifacts/evaluation/direction_history_2011 --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_history_direction
python tools/evaluate_core_direction_engines.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_core_direction_engines
```

## 恒生数据时点核查

正式快照的 `raw/hangseng.csv` 已包含 2026-09-17 行情；工作目录中较旧的行情文件不能代表已发布快照的实际输入。以该归档冻结输入，分别测试了上一已完成港股交易日、当日港股价格及保留原特征并追加当日价格三种方案，结果位于 `artifacts/evaluation/hangseng_timing_v1/`。当日价格方案以北京时间 18:00 发布为前提，成交量始终使用前一已完成港股交易日。

| 方案 | 2023 年准确率 | 2024 年准确率 | 开发期筛选 |
| --- | ---: | ---: | --- |
| 原模型 | 48.35% | 54.55% | 对照 |
| 上一已完成港股交易日 | 47.11% | 54.13% | 不通过 |
| 当日港股价格 | 48.76% | 52.89% | 不通过 |
| 追加当日港股价格 | 48.76% | 52.07% | 不通过 |

三种方案均未同时通过两个年度的准确率和平衡准确率要求，因此没有进入后续回归验收，也没有调整生产数据时点。时点测试覆盖港股休市日、同日价格的可用时间限制和未来行情隔离。

## 海外风险输入

增加标普 500 和纳斯达克历史收盘价作为独立市场信息，完整数据冻结于 `artifacts/evaluation/global_risk_data_20260917_complete/`，两个指数均有 1,695 行，覆盖 2019-12-16 至 2026-09-15。相同北京时间日期的美股收盘尚未发生，因此每个国内信号日只允许使用美股日期严格早于该日的记录。新增特征为各指数的 1 日、5 日收益、20 日波动率和回撤，以及纳斯达克相对标普的当日收益。

使用现有小型误差逻辑回归，加入上述特征；冻结 756 日训练窗口、至少 252 个已结算样本、每 5 日重训和 60% 反向置信门槛。只比较普通权重和按实际涨跌平衡训练权重两个候选，未按近期连涨日期选择参数。

| 候选 | 2023 年准确率 | 2024 年准确率 | 开发期筛选 |
| --- | ---: | ---: | --- |
| 原模型 | 48.35% | 54.55% | 对照 |
| 海外风险误差模型 | 50.00% | 54.13% | 不通过 |
| 海外风险误差模型，平衡权重 | 50.00% | 52.48% | 不通过 |

两者均未通过 2024 年准确率和平衡准确率要求，因此止于开发期，没有运行 2025 年以后回归区间来挑选候选。结果与冻结源码位于 `artifacts/evaluation/global_risk_error_v1/`，`report.json` 为 `passed: false`。首次获取中途中断和第二次日期格式校验失败的目录不属于有效输入；仅使用带有完整 `manifest.json` 的 `global_risk_data_20260917_complete/`。

相关测试覆盖美股当日收盘排除、未来行情和当前结果隔离、两资产日期覆盖不一致、陈旧或重复数据拒绝、非默认行索引对齐，以及两个候选的完整历史和前缀回放逐字段相等。历史日线数据没有逐笔发布和修订时间戳，因而这些检查不构成实时数据送达保证。

本轮 55 项相关测试通过，语法检查与按仓库原有换行配置执行的 `git diff --check` 通过。修复了非默认行索引的对齐后，重新生成的全部海外特征与本轮冻结输入逐字段精确相等。正式模型四个核心文件的哈希仍与原始纠偏对照合同一致，9 月 17 日归档 `results.csv` 的 SHA256 仍为 `49d889325b8a44a5de4e3aa9ab44aa31d787b1d54a207be8a6eea2652abe0d86`。

运行态核查中，浏览器原地址的 8010 端口没有监听；8005 服务返回的快照为 `7d7c044b-9c01-4aff-ad72-92393b7f85d2`，行情截至 2026-09-15，CSV 哈希为 `4649665bc518655758174fe1a7591f977f10a6b2f44c6769198085d8cd7a97ef`。这是另一份运行态输出，不能据此声称它与上述 9 月 17 日归档相同；本轮没有重启该进程或更换其发布数据。

```powershell
python tools/fetch_global_risk.py --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_global_risk_data
python tools/evaluate_directional_error.py --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --global-context artifacts/run/new_global_risk_data --output artifacts/run/new_global_risk_error
```

## 规则投票上下文与改判门槛

进一步检查规则意见是否集中、是否主要由慢变特征提供、以及同方向持续时间是否有助于识别错误。新增投票差距、有效专家数、慢变规则权重、收盘位置规则权重、外部规则权重、过去 20 日规则持续性、相对多数方向基准的历史优势、方向持续长度、状态否决和失败保护标记。所有排名和权重从冻结行情重新计算，与 1,397 条原始诊断的基础方向逐日核对一致；不使用诊断中的当日实际涨跌或命中结果。

单一候选为带方向交互项的逻辑回归误差模型，固定 756 日训练窗口、至少 252 行历史、每 5 日重训、`C=0.1`。默认 60% 改判门槛在 2023 年准确率持平但平衡准确率下降，2024 年准确率为 53.72%，低于基线 54.55%，见 `artifacts/evaluation/rule_context_error_v1/`。

随后仅在开发期选择此前已使用的 55%、60%、65%、70%、75% 五个门槛。规则上下文共 5 个组合，海外风险两个候选共 10 个组合，全部尝试和筛选依据保存在各自合同及 `selection.json` 中。每个系列只让开发期选中的一个候选进入后续回归，未按回归期结果改选另一个门槛。

| 实验 | 开发期选中门槛 | 415 日准确率 | 近 60 日准确率 | 近 60 日下跌召回率 | 近 60 日最长看涨 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原模型 | 无 | 58.07% | 58.33% | 55.56% | 15 |
| `rule_context_selective_v1` | 55% | 55.42% | 60.00% | 48.15% | 15 |
| `global_risk_selective_v1` | 65% | 58.55% | 63.33% | 44.44% | 15 |

规则上下文候选在回归期修正 29 次错误、新增 40 次错误。海外风险候选修正 6 次错误、新增 4 次错误，但减少了下跌识别，近 60 日看涨数量由 32 次增至 41 次；近 20 日仍为 18 次看涨，且幅度 MAE、RMSE 上升。两者都未通过全部纠偏和非回退条件，没有接入生产。

新增测试验证排名重建与生产投票一致、当前及未来标签隔离、前缀回放、原预测不可变、源日期对齐和改判门槛必须使用原始概率重新构造提议方向。测试用短行情序列显式配置较短的最小历史，生产配置没有调整。

```powershell
python tools/evaluate_directional_error.py --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --rule-run artifacts/evaluation/direction_guard_v2 --threshold-grid --output artifacts/run/new_rule_context_selective
python tools/evaluate_directional_error.py --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --global-context artifacts/evaluation/global_risk_data_20260917_complete --threshold-grid --output artifacts/run/new_global_risk_selective
```

## 沪深市场广度

新增逐日实际成交股票的广度信息。`market_breadth_20260917_v2/` 冻结了 2020-11-17 至 2026-09-15 的 1,416 个交易日，共 7,209,073 条原始日线记录；压缩原始响应约 139 MB，每日文件都有 SHA256。以接口返回的当日沪深成交股票为分母，没有拿今天的上市股票名单筛选历史。每天有效沪深样本为 4,063 至 5,209 只。

首次获取在非有限涨跌幅校验处中止，原因是接口同时返回了北交所成立前的场外交易历史。随后固定统计范围为沪深交易所，排除所有 `.BJ` 记录，并保留排除数量；原始响应仍完整保存。停牌或零成交记录不进入有效样本，缺失涨跌幅不能静默视为平盘。原始记录使用分页请求获取，重复证券、异常日期和不完整历史会被拒绝。

新增指标分别统计沪深整体及上交所的上涨家数比例、下跌家数比例、收益均值与中位数、下跌股票成交金额占比、跌幅至少 2% 的家数比例、上涨比例的一日变化和 5/20 日均值，以及上交所相对沪深整体的广度差。候选沿用既定浅层 XGBoost 设置，比较普通方向、平衡方向和误差预测三个模型，每个模型仅使用之前已固定的五个改判门槛。

第一轮把所有广度指标滞后一个国内交易日。15 个组合均未通过 2023、2024 年开发期筛选，见 `breadth_direction_v1/`；高门槛组合因几乎没有改判，也未被当作纠偏成功。

随后核实 [Tushare 日线说明](https://tushare.pro/document/2?doc_id=27) 表明数据在交易日 15:00–16:00 入库，本项目配置的定时刷新时间为 18:15，因此增加当日收盘广度方案。`market_breadth_close_20260917/` 复用了并校验原有 1,416 份响应，只补取 2026-09-16 一日，共 1,417 日、7,214,623 条记录。两个数据集前 1,416 行广度结果逐字段精确相等，原有滞后一日特征也与旧冻结实现的 1,397 行结果精确一致。

当日方案仅允许在北京时间 17:00 以后生成，实验声明为 18:00；原始历史响应没有逐条入库时间，因此按官方入库规则进行时点假设，不将其表述为实际送达时间的逐日证明。特征计算和模型训练均通过前缀回放及未来数据隔离测试。

当日方案开发期仅 `breadth_direction` 的 55% 改判门槛合格，2023 年准确率从 48.35% 升至 50.83%，2024 年准确率持平于 54.55%。只让这一个开发期选中的组合进入后续回归，结果位于 `breadth_close_direction_v1/`：

| 范围 | 原模型准确率 | 候选准确率 | 结论 |
| --- | ---: | ---: | --- |
| 415 日回归期 | 58.07% | 56.87% | 回退 |
| 近 252 日 | 58.73% | 58.33% | 回退 |
| 近 60 日 | 58.33% | 58.33% | 方向未改善 |
| 近 20 日 | 60.00% | 60.00% | 仍有 18 次看涨 |

回归期修正 9 次错误、新增 14 次错误，近 60 日最长连续看涨仍为 15 次。虽然回归期整体的下跌召回率有所提高，但近期集中没有缓解，且准确率和置信度校准等指标回退，未发布。22 项广度与残差模型相关测试通过，覆盖暂停交易样本的处理、百分数单位、成交金额权重、分页、缓存校验、两种时点策略、当前/未来结果隔离和原预测不变。

复现时使用新的输出目录，原始数据的 `--resume` 只恢复同一冻结合同；完整数据的 `--reuse-from` 会逐文件验哈希并复用原始响应，避免再次请求历史数据：

```powershell
python tools/evaluate_breadth_direction.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --context artifacts/evaluation/direction_context_20260917 --breadth artifacts/evaluation/market_breadth_20260917_v2 --output artifacts/run/new_breadth_direction
python tools/evaluate_breadth_direction.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --context artifacts/evaluation/direction_context_20260917 --breadth artifacts/evaluation/market_breadth_close_20260917 --lag-sessions 0 --publication-hour 18 --output artifacts/run/new_breadth_close_direction
```

## 条件下跌专用模型

前述双向误差模型可能通过把看跌改为看涨来提高总体准确率。例如，海外风险选择模型在最近 60 日的 9 次改判全部是看跌改看涨。因此新增一个独立假设：只用原模型历史看涨且已经结算的样本学习下跌概率，保持所有原始看跌方向。此设计限制纠正方向，但不按连续看涨天数触发，也不要求固定涨跌比例。

`downside_specialist_v1/` 冻结了两种模型：`C=0.1` 的逻辑回归和深度为 2 的浅层 XGBoost。共同使用过去 756 个信号日内至少 120 条已结算看涨样本，每隔至少 5 个信号日重训。输入为原始及校准置信度、6 个价格指标、6 个当日收盘广度指标和 4 个上一美股交易日风险指标，共 18 项。数据可用时间假设沿用前述广度和海外数据约束。

仍只在 2023、2024 年按原有条件选择 55%、60%、65%、70%、75% 五个门槛，共 10 个组合。浅层树没有通过开发期的组合；逻辑回归有 3 个合格门槛，按预先冻结的排序选中 55%。其开发期两年准确率分别为 49.59%、55.37%，均高于基线 48.35%、54.55%。随后只对该组合运行完整回归，没有根据回归结果改选门槛。

| 范围 | 原模型准确率 | 候选准确率 | 原模型下跌召回率 | 候选下跌召回率 |
| --- | ---: | ---: | ---: | ---: |
| 415 日回归期 | 58.07% | 56.87% | 46.41% | 60.77% |
| 近 252 日 | 58.73% | 59.13% | 45.61% | 49.12% |
| 近 60 日 | 58.33% | 60.00% | 55.56% | 59.26% |
| 近 20 日 | 60.00% | 60.00% | 20.00% | 20.00% |

该候选在回归期修正 26 次错误、新增 31 次错误；尽管下跌召回率上升，整体准确率下降。近 20 日仍为 18 次看涨，近 60 日最长连续看涨仍为 15 次。26 项验收有 8 项未通过，包括回归准确率、回归 MAE、四个窗口的置信度 Brier 和两项近期纠偏条件。结论仍为不发布，不能将近 60 日多命中 1 次表述为完成纠偏。

新增 8 项测试全部通过，验证两个真实模型的前缀回放、当前及未来结果隔离、训练仅使用已结算看涨样本、原始看跌方向保留、未来行情隔离、数据缺失拒绝及门槛处理。完整回放与开发期前缀预测逐字段精确相等。本轮再次核对，正式四个核心源码文件和 9 月 17 日冻结归档 CSV 的哈希未变；没有修改 Web 面板或替换运行中服务的数据。

```powershell
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --output artifacts/run/new_downside_specialist
```

目前这些结果支持的结论是：已测试的方法没有同时满足用户的两个要求，而不是证明任何方法都不可能实现。继续围绕同一次连涨区间调整模型会增加过拟合风险；后续需要有独立依据的新假设，并用新的事前预测记录检验效果。

## 分段规则验证

进一步复核原规则的阈值、标签和置信度校准时点，没有发现导致此前候选失败的未来数据读取。原规则仍存在一个统计问题：同一段历史同时用于规则排名、方向取反和成绩估计，被选中规则的表观成绩可能高估其后续能力。

`validated_rules_v1/` 针对此问题冻结两个候选。两者均先在较早的 120 个信号日学习规则方向并选出最多 5 条不同路径，再用后续 60 个已结算信号日检查涨跌区分能力；后段的每一类至少 10 行，平衡准确率须大于 50%。后段不能重新取反失败的规则，投票权重仍来自前段。无规则合格时使用原有选择器。两者仅前段排名指标不同，分别为准确率和平衡准确率。

| 模型 | 2023 年准确率 | 2024 年准确率 | 开发期改判次数 |
| --- | ---: | ---: | ---: |
| 原模型 | 48.35% | 54.55% | 0 |
| 分段验证，准确率排名 | 46.28% | 47.11% | 179 |
| 分段验证，平衡准确率排名 | 47.93% | 49.59% | 165 |

两个候选均未通过开发期，未运行 2025 年以后的比较来另选参数。8 项测试通过，覆盖两段不重叠、排名不依赖验证结果、验证失败后禁止重新取反、常量多数方向不能视为有区分能力、互补路径去重、完整信号的前缀回放和未来数据隔离。

```powershell
python tools/evaluate_core_direction_engines.py --family validated-rules --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_validated_rules
```

## 股指期货基差与持仓

增加沪深 300 和中证 500 对应的 IF、IC 股指期货，检验对冲需求与期现价格分歧能否提供独立下跌信息。Tushare `fut_daily` 接口已验证可用；数据冻结在 `futures_context_20260917_v2/`，覆盖 2020-11-17 至 2026-09-15 的 1,416 个交易日、71 个月。保存 82,806 条中金所原始日线响应，其中用于计算的历史实际 IF/IC 月合约共 11,328 条，每产品每天恰有 4 个不同交割月份的合约。

连续与主力拼接代码（如 `IF.CFX`）不参与计算。以各日成交量选择实际月合约，其收益和基差变化使用该合约自身的 `pre_close`，避免将换月价差当成收益；总持仓和总成交量汇总该日全部 4 个月合约。部分非主力合约的昨收盘价为空，相关字段不用于其持仓或基差汇总，也不填补；如选中的主力合约缺少昨收盘，直接拒绝计算。最初 `futures_context_20260917/` 因此项校验中止，原始月份文件逐一验哈希复用于有效版本，没有重新请求全部历史。

每个产品新增 8 个指标：主力基差、同合约基差变化、持仓加权基差、近两月合约每月期限价差、成交量与持仓比、相对现货收益、20 日成交量比和总持仓变化。全部指标严格滞后一个国内交易日，无中间日期填补。原始日线仍没有逐条历史发布和修订时间戳。

`futures_downside_v1/` 在上一轮专用模型的 18 项输入上增加上述 16 项期货指标，两个模型的训练窗口、正则化和五个改判门槛均不改变。开发期只有浅层树的 60% 门槛通过筛选，2023、2024 年准确率分别为 48.76%、55.79%，开发期改判 20 次。仅此组合进入后续回归：

| 范围 | 原模型准确率 | 候选准确率 | 结果 |
| --- | ---: | ---: | --- |
| 415 日回归期 | 58.07% | 57.59% | 新增 2 次错误，无修正 |
| 近 252 日 | 58.73% | 58.73% | 方向未改变 |
| 近 60 日 | 58.33% | 58.33% | 最长连续看涨仍为 15 次 |
| 近 20 日 | 60.00% | 60.00% | 仍有 18 次看涨 |

26 项验收有 8 项失败，包括回归准确率、平衡准确率、幅度 MAE/RMSE、回归及近 252 日 Brier，以及两项近期纠偏条件，未发布。完整回放的开发期前缀与冻结预测逐字段精确相等。

期货数据的 11 项测试通过，覆盖真实月合约选择、换月、总持仓、无关缺失字段、主力昨收盘缺失拒绝、分页、错误信息去敏、前一交易日对齐、缺失交易日拒绝与未来数据隔离。加上本轮分段规则的 8 项测试，共 19 项新增测试通过。正式四个核心源码文件和 9 月 17 日归档 CSV 的哈希再次核对未变，没有改动面板或运行中的发布数据。

```powershell
python tools/fetch_futures_context.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --context artifacts/evaluation/direction_context_20260917 --output artifacts/run/new_futures_context
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --futures-context artifacts/evaluation/futures_context_20260917_v2 --output artifacts/run/new_futures_downside
```

## 当日期货与价格方向规则

追溯 Git 历史发现，当前的 `state_veto_window=75`、`state_veto_bad_accuracy=0.46`、120 日规则校准和 5 条规则投票，在 2026-07-10 的 `9266c7a` 提交中已经存在。这不能证明参数最初的选择过程未接触其他历史数据，但没有证据表明它们是针对 8 月 19 日至 9 月 8 日这次连续看涨事后设置的。9 月的固定校准改造仍要求冻结基线的方向逐日一致。

考虑到实际发布在收盘后，新增当日期货方案，假设当日收盘行情在北京时间 18:00 可用；这是一项历史回放假设，没有逐日原始送达时间戳。`futures_close_context_20260917/` 复用前 70 个月响应，仅重新获取扩展后的 2026 年 9 月数据，截至 2026-09-16 共 1,417 个交易日、82,870 条中金所原始日线。前 1,416 日汇总与原版逐字段一致，默认滞后一日逻辑的全部 1,397 行特征与旧冻结源码精确一致。

模型参数和门槛均不变。`futures_close_downside_v1/` 开发期选出逻辑回归及 60% 门槛，2023、2024 年准确率分别为 49.17%、54.55%。回归期修正 7 次错误、新增 11 次错误，准确率为 57.11%，低于基线 58.07%；近 252 日为 58.33%，低于 58.73%。近 60/20 日方向未改，最长连续看涨仍为 15 次，26 项验收有 12 项失败，未发布。

另一个固定候选 `current_price_rules_v1/` 针对规则候选白名单没有使用当日直接价格方向的问题，只追加 `return_1`、`overnight_gap` 和 `intraday_return` 三项已有价格指标，沿用原有历史分位数、准确率排名、投票、状态否决和校准，不修改原规则权重。其 2023 年准确率为 48.76%，但 2024 年降至 50.00%，开发期改判 88 次，未通过筛选，未继续用近期数据选择参数。

本轮期货相关 13 项测试和新增价格规则 3 项测试通过；测试涵盖两种发布时间假设、前缀一致性、换月、互补规则、当前结果及未来数据隔离。特征预热测试明确区分收益首日为空时需要多一行历史，不修改生产参数。

```powershell
python tools/fetch_futures_context.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --context artifacts/evaluation/direction_context_20260917 --reuse-from artifacts/evaluation/futures_context_20260917_v2 --include-current --output artifacts/run/new_futures_close_context
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --futures-context artifacts/evaluation/futures_close_context_20260917 --futures-lag-sessions 0 --publication-hour 18 --output artifacts/run/new_futures_close_downside
python tools/evaluate_core_direction_engines.py --family current-price --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_current_price_rules
```

## 上证 50ETF 期权需求

为检验期权市场的下跌对冲需求是否包含额外信息，新增 Tushare `opt_basic` 和 `opt_daily` 数据。固定标的为 `OP510050.SH`，使用 5,040 份历史合约信息，包括已到期合约，最早上市日为 2015-02-09。逐日以开始交易日和最后交易日核对在市合约，实际日线必须完整覆盖当日全部合约，缺失不能当作零成交。

完整输入位于 `option_context_20260917/`：2020-11-17 至 2026-09-15 共 1,416 个交易日，保留 71 个月的上交所原始日线响应，共 655,652 行、压缩后约 8.8 MB，其中上证 50ETF 期权为 187,610 行。使用认沽成交量、成交金额和持仓量分别占认沽加认购总量的比例，以及各比例的一日变化和 20 日均值，共 9 项特征。所有特征滞后一个国内交易日，既不使用事后调整的行权价格和合约乘数，也不据此假定历史原始数据不存在修订。

`option_downside_v1/` 将这 9 项指标加入原来的 18 项专用模型输入，沿用已冻结的两个模型参数和五个改判门槛。开发期只有逻辑回归存在合格门槛，按既定排序选中 60%；2023、2024 年准确率分别为 49.59%、56.61%，开发期改判 88 次。

| 范围 | 原模型准确率 | 候选准确率 | 原模型下跌召回率 | 候选下跌召回率 |
| --- | ---: | ---: | ---: | ---: |
| 415 日回归期 | 58.07% | 57.59% | 46.41% | 55.80% |
| 近 252 日 | 58.73% | 57.54% | 45.61% | 48.25% |
| 近 60 日 | 58.33% | 58.33% | 55.56% | 59.26% |
| 近 20 日 | 60.00% | 60.00% | 20.00% | 20.00% |

回归期修正 17 次错误、新增 19 次错误，26 项验收有 12 项失败。近 20 日依然 18 次看涨，近 60 日最长连续看涨仍为 15 次，未发布。完整回放的开发期前缀与冻结预测逐字段相等。

期权相关 12 项测试通过，覆盖在市合约变化、到期合约保留、零成交合约计入、缺失和重复合约拒绝、比例单位、分页、错误去敏、严格前一交易日对齐和未来数据隔离。不能用当前仍在交易的合约名单替代本次冻结的历史合约目录。

```powershell
python tools/fetch_option_context.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_option_context
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --option-context artifacts/evaluation/option_context_20260917 --output artifacts/run/new_option_downside
```

## 规则相关系数评分

新增单一候选 `correlation_rules_v1/`，用马修斯相关系数（MCC）替代总体准确率排名。常量预测相关系数为零，不参与当前投票；负相关规则按历史相关符号取反，仍取前 5 条规则并保留重复路径的累计排名权重。校准窗口、阈值、状态否决与置信度和幅度校准均保持原设置。没有可用非零相关规则时使用原选择器。

开发期 2023 年准确率为 50.83%，高于原模型 48.35%；但 2024 年为 53.31%，低于 54.55%，且平衡准确率也下降，因此未通过两个年度分别不回退的筛选。共改判 153 次，没有进入后续回归阶段。4 项测试通过，向量化 MCC 与 scikit-learn 的标准实现一致，并覆盖缺失规则、常量方向、负相关取反和完整信号的前缀一致性。

```powershell
python tools/evaluate_core_direction_engines.py --family correlation --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_correlation_rules
```

本轮共 16 项新增测试通过，正式四个核心文件与 9 月 17 日冻结归档 CSV 哈希保持不变，面板和运行中服务数据没有因本轮实验而修改。两项实验都不构成完成纠偏或保证未来性能不回退的证据。

## 条件下跌概率校准

`calibrated_option_downside_v1/` 检验辅助模型的原始下跌概率是否低估了改判机会。在上一轮期权专用模型上增加单调保序校准，使用此前 252 个信号日内至少 60 条已结算、原模型看涨且辅助模型已训练的预测。当前及未来结果、未结算结果、原始看跌预测均不进入校准样本；样本不足或只有一种实际方向时保留原概率。两个辅助模型的训练参数与既有五个改判门槛不变。

开发期只有逻辑回归的 55% 门槛合格，2023、2024 年准确率分别为 50.00%、56.20%，开发期改判 110 次。只对这个选中的组合执行完整回归；改判后仍用自身此前的混合预测进行既有幅度和置信度校准。

| 范围 | 原模型准确率 | 候选准确率 | 候选下跌召回率 |
| --- | ---: | ---: | ---: |
| 415 日回归期 | 58.07% | 58.07% | 50.83% |
| 近 252 日 | 58.73% | 59.13% | 46.49% |
| 近 60 日 | 58.33% | 60.00% | 59.26% |
| 近 20 日 | 60.00% | 60.00% | 20.00% |

回归期修正 8 次错误、新增 8 次错误。虽然四个窗口的方向准确率、平衡准确率和下跌召回率都没有下降，近 20 日仍有 18 次看涨，近 60 日最长连续看涨仍为 15 次。26 项验收有 6 项失败：回归期的置信度 Brier，近 20 日的收益 MAE、收益 RMSE、置信度 Brier，以及近 20 日上涨比例差和近 60 日最长连续看涨。该候选未通过验收，没有发布；不能将近 60 日多命中一次称为解决看涨偏向。

完整回放的开发期前缀与冻结预测逐字段相等。8 项校准测试通过，覆盖当前及未来结果隔离、未训练或未结算样本排除、源预测不变、无效概率与训练日期拒绝，以及校准、改判和最终预测字段共同回放的一致性。最后一项验证同时检查方向、预测收益、收盘价、原始及校准置信度、下跌概率和校准日期，并确认原始看跌方向保留。

另外复核 `rule_context_selective_v1/rule_features.csv`：1,397 个信号日仅 4 次规则投票差接近零，近 60 日没有平票。连续 15 次看涨期间最小投票差为 0.12328767123287698，因此平票默认看涨不是这次集中现象的原因。

本轮再次核对，17 个实验源码哈希均与冻结合同一致，正式四个核心源码及 9 月 17 日归档 CSV 哈希也未变化。未修改面板或运行中服务的发布数据。

```powershell
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --option-context artifacts/evaluation/option_context_20260917 --calibrate-downside --output artifacts/run/new_calibrated_option_downside
python -m pytest tests/test_downside_probability_calibration.py -q
```

## 新增银行间与交易所资金压力信号

用户授权继续调用 Tushare 获取新信号后，新增 SHIBOR、上交所 GC001（`204001.SH`）和 GC007（`204007.SH`）历史数据，覆盖 2020-11-17 至 2026-09-15。按年请求并冻结原始响应，SHIBOR 共 1,445 条，两个回购品种各 1,416 条。13 项新特征包括隔夜利率及其 1/5 日变化、相对 20 日均值的变化、银行间期限利差、交易所与银行间利差、回购期限利差、日内利率范围和成交量变化；全部严格滞后一个国内交易日。

首次完整性检查发现两个回购品种均缺少 2021-04-23，且各含 2021-10-06 休市日记录。对两个日期分别再请求单日数据，缺失日仍为空，休市日仍有记录。未手动改日期，也没有用旧值填补。初次输出 `liquidity_context_20260918/` 保留为未完成的获取记录；有效数据集 `liquidity_context_20260918_v2/` 逐文件验证并复用 21 份原始响应，按冻结股票交易日对齐，列明排除日期和缺口。

缺口及 20 日滚动窗口导致 2021-04-26 至 2021-05-26 的 20 个信号日特征不可用。辅助模型明确排除这些行的训练和改判，原方向保留；这些行仍参与最终同日成绩比较。SHIBOR 在部分股票休市的补班工作日有报价，这些额外日期也不进入国内交易日滚动窗口。历史响应不带原始发布或修订时间戳，滞后一日是研究时点约束，不是实际送达时间的证明。

`liquidity_downside_v1/` 在原专用模型的 18 项输入上增加上述 13 项信号，训练参数与五个改判门槛保持原设置。开发期仅逻辑回归 70% 门槛合格，2023、2024 年准确率分别为 49.17%、55.37%，改判 16 次；只让该组合进入回归。

| 范围 | 原模型准确率 | 候选准确率 | 候选下跌召回率 |
| --- | ---: | ---: | ---: |
| 415 日回归期 | 58.07% | 58.31% | 46.96% |
| 近 252 日 | 58.73% | 59.13% | 46.49% |
| 近 60 日 | 58.33% | 60.00% | 59.26% |
| 近 20 日 | 60.00% | 60.00% | 20.00% |

回归期修正 1 次错误，未新增方向错误，但近 20 日仍为 18 次看涨，近 60 日最长看涨仍为 15 次。26 项验收失败 4 项：近 20 日的收益 MAE、RMSE，以及两项近期纠偏条件。因此未发布。完整回放的开发期前缀逐字段一致。

新增 13 项资金特征测试通过，包括两个真实模型在缺失特征下的历史样本排除与前缀回放、概率回退、日期对齐、单位、滚动窗口、异常数据拒绝及请求错误去敏；加上原专用模型和概率校准，共 29 项测试通过。正式核心源码和 9 月 17 日归档 CSV 哈希未改变。

```powershell
python tools/fetch_liquidity_context.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_liquidity_context
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --liquidity-context artifacts/run/new_liquidity_context --output artifacts/run/new_liquidity_downside
```

## 个股分布尾部、涨跌转换与成交集中度

第二个独立假设是，指数方向和上涨家数比例未充分表达个股下跌扩散及交易集中度。复用 `market_breadth_close_20260917/` 中逐文件验哈希的 1,417 份 Tushare 日线响应，共 7,214,623 行，不再次请求相同数据。新数据集 `distribution_context_20260918/` 保存 1,416 日相邻日期统计及覆盖全部 1,397 个基线信号日的特征。

新增 10 项特征：当日收益的 10% 和 90% 分位数、四分位距、成交金额加权收益与等权收益之差、成交金额最大 10% 股票的成交占比、昨日涨今日跌比例、连续下跌比例、昨日跌今日涨比例，以及收益左尾的一日变化和由涨转跌比例的 5 日均值。金额加权收益的计算仅对当日横截面的 1%/99% 极端收益截尾；分位数本身不截尾。转换比例以相邻两日均有实际成交的匹配股票为分母，不使用今天的上市名单筛选历史，也不把停牌或新上市股票算作方向转换。

数据时间约束沿用官方日线入库说明，实验假定北京时间 18:00 发布；只有完成的当日横截面及之前日期参与计算。该假定不构成历史实际数据送达证明。此实验独立于资金压力实验，没有读取回归表现后把两组信号组合选择。

`distribution_downside_v1/` 沿用两个专用模型及五个改判门槛。开发期选中浅层 XGBoost 的 55% 门槛，2023、2024 年准确率分别为 50.00%、55.79%，改判 53 次。

| 范围 | 原模型准确率 | 候选准确率 | 候选下跌召回率 |
| --- | ---: | ---: | ---: |
| 415 日回归期 | 58.07% | 55.66% | 57.46% |
| 近 252 日 | 58.73% | 57.94% | 55.26% |
| 近 60 日 | 58.33% | 58.33% | 55.56% |
| 近 20 日 | 60.00% | 60.00% | 20.00% |

回归期修正 20 次错误、新增 30 次错误；近 20 日仍为 18 次看涨，近 60 日最长连续看涨仍为 15 次。26 项验收有 12 项失败：回归准确率、平衡准确率、收益 MAE/RMSE、置信度 Brier，近 252 日准确率、收益 MAE、置信度 Brier，近 60 日和近 20 日置信度 Brier，以及两项近期纠偏条件。没有发布该候选，也未改选其他门槛来迎合回归结果。

完整回放的开发期前缀一致。新增 8 项测试验证匹配股票分母、收益单位、停牌及新股处理、源日期和发布时点、缺失数据拒绝与前缀回放。加上本轮资金特征及既有两个专用模型相关测试，共 37 项通过。测试通过说明实现满足这些检查，不能代替预测效果验收。

```powershell
python tools/build_distribution_context.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --output artifacts/run/new_distribution_context
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --distribution-context artifacts/run/new_distribution_context --output artifacts/run/new_distribution_downside
python -m pytest tests/test_liquidity_features.py tests/test_distribution_features.py tests/test_downside_specialist.py tests/test_downside_probability_calibration.py -q
```

## 逐日预测成绩排名

`prequential_rules_v1/` 检验原规则评分中的另一个统计假设。原评分在今天决定规则方向后，会用这个方向重新评估过去 120 日。新候选先对每个历史日只用当时此前的结果决定是否取反，保存那一天实际能够给出的规则方向，再用这些方向后续结算的成绩排名。今天选出的规则不能改写它们以前的方向；集合置信度也来自以前各日按当时排名生成的集合方向，而不是把今天的排名套用到旧数据。

冻结两个候选，分别按普通准确率和平衡准确率排名。规则阈值、120 日窗口、最小 30 行、前 5 个排名名额和重复路径权重保持原设置。只有过去逐日预测的评分超过 50% 的规则进入投票；平衡评分还要求两类历史各至少 10 行。没有合格规则时使用原选择器。候选方向开始产生后，在其集合方向尚不足 30 条已结算记录时，原始置信度按 50% 处理，不把规则选择期的成绩当成集合预测成绩。

| 候选 | 2023 年准确率 | 2024 年准确率 | 开发期改判 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原模型 | 48.35% | 54.55% | 无 | 对照 |
| 逐日预测成绩，普通准确率 | 49.17% | 51.65% | 127 | 不通过 |
| 逐日预测成绩，平衡准确率 | 44.21% | 49.59% | 148 | 不通过 |

两个候选均未通过两个开发年份分别不回退的要求，没有进入 2025 年后的回归验收，也没有接入生产。该结果不能证明原评分方式没有统计偏差，只说明这个更改在现有规则上没有给出合格替代模型。

10 项测试通过，覆盖滚动窗口排除当前结果、规则方向不被后来结果改写、缺失已结算样本排除、两种排名的前缀一致性与未来结果隔离、常量看涨规则不获平衡评分优势、重复路径权重、候选集合自身的历史校准及初期置信度处理。

```powershell
python tools/evaluate_core_direction_engines.py --family prequential --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_prequential_rules
python -m pytest tests/test_prequential_rule_candidates.py -q
```

## 个股订单金额分组买卖差

新增假设是，按成交金额分类的大单卖出压力和大小单分歧，可能提供价格、波动率及上涨家数之外的下跌信息。`moneyflow_context_20260918/` 获取 2020-11-17 至 2026-09-15 的 Tushare `moneyflow` 历史响应，逐日保存、分页、验哈希并支持同一合同中断续传；同时复用已冻结的个股日线，按当日真实有成交的沪深股票核对覆盖率。

[Tushare 接口说明](https://tushare.pro/document/2?doc_id=170) 将小单定义为 5 万元以下、中单为 5 万至 20 万、大单为 20 万至 100 万、特大单为至少 100 万元，金额字段单位为万元。这里计算大单与特大单的买入金额减卖出金额，以及小单的对应差值；分母为匹配股票四类买卖金额之和。这是订单金额分组的买卖差，不是投资者身份或机构持仓，也不等于另有计算规则的 `net_mf_amount` 字段。

新增 10 项特征：大单买卖差比例、小单买卖差比例、大单卖出占优的股票比例及金额覆盖面、股票大单买卖差比例的中位数、大单卖出且小单买入的分歧比例、上交所相对沪深整体的大单买卖差，以及大单买卖差的一日变化和 5/20 日均值。所有特征滞后一个国内交易日。历史响应没有逐条发布或修订时间，不能用这个滞后假定声称已证明历史实际送达时间。

每日股票数量与成交金额覆盖率都要求达到 98%。低覆盖记录保留原始响应，统计值及受影响的滚动窗口标为不可用；不将缺失流向填成零，也不用旧值填补。辅助模型只训练在特征完整且此前已结算的样本上，当日特征不可用则保留原方向。模型训练参数、两个候选、五个改判门槛、开发年份和 26 项回归条件沿用既有专用模型设置。

本轮新增 11 项测试通过，覆盖金额差符号、金额权重、停牌及非沪深样本排除、低覆盖处理、高股票覆盖但低成交额覆盖的拒绝、无效源记录、数据滞后和前缀回放、缺失覆盖向滚动窗口传播、分页与错误去敏。加上既有两个专用模型相关测试，共 27 项通过。

数据获取已完成：1,416 个交易日、6,928,930 条原始记录，压缩原始响应约 242.5 MiB。最低股票覆盖率为 99.84%，最低成交金额覆盖率为 99.11%，全部 1,397 个基线信号日都有可用特征。再次核验了全部 1,416 份原始文件哈希，抽取三个时期重新聚合与冻结 CSV 精确一致，真实特征在三个截断日期的前缀回放也逐字段一致。

`moneyflow_downside_v1/` 在原专用模型的 18 项输入上加入这 10 项特征。仅逻辑回归有开发期合格门槛，按既定排序选中 60%；2023、2024 年准确率分别为 50.41%、54.96%，开发期改判 58 次。仅这个组合进入完整回归。

| 范围 | 原模型准确率 | 候选准确率 | 候选下跌召回率 |
| --- | ---: | ---: | ---: |
| 415 日回归期 | 58.07% | 58.55% | 50.28% |
| 近 252 日 | 58.73% | 58.73% | 46.49% |
| 近 60 日 | 58.33% | 60.00% | 59.26% |
| 近 20 日 | 60.00% | 65.00% | 30.00% |

回归期修正 7 次错误、新增 5 次错误。四个窗口的准确率、平衡准确率、下跌召回率、收益 MAE/RMSE、置信度 Brier 均通过非回退检查。近 20 日看涨次数从 18 次降为 17 次，实际上涨仍为 10 次，上涨占比差从 40 个百分点缩小至 35 个百分点；但近 60 日最长连续看涨仍为 15 次。因此 26 项通过 25 项，失败项是近 60 日最长连续看涨，尚未满足全部纠偏条件，未发布。

完整回放与开发期前缀精确一致，生产四个核心源码与已发布归档哈希保持不变。本轮改善只是在已观察回归数据中多命中两次，不能据此宣称已解决连续看涨或能保证未来预测不回退。

```powershell
python tools/fetch_moneyflow_context.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --output artifacts/run/new_moneyflow_context
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --moneyflow-context artifacts/run/new_moneyflow_context --output artifacts/run/new_moneyflow_downside
```

## 资金流向概率校准与联合输入

在不查看连涨区间逐日标签来调整参数的前提下，先检查资金流向模型的开发期概率。逻辑回归给出平均 30.62% 下跌概率的 59 条预测，实际下跌为 44.07%；高于 70% 的 16 条预测平均概率为 74.40%，实际下跌为 43.75%。这是开发期的概率可靠性检查，不能当作未来误差保证。

`calibrated_moneyflow_downside_v1/` 沿用此前冻结的单调保序校准：过去 252 个信号日、至少 60 条已训练且已结算的原模型看涨样本。模型和五个门槛保持原设置，开发期选中逻辑回归 75% 门槛，2023、2024 年准确率分别为 48.76%、54.55%，改判 9 次。回归期没有改变任何方向。26 项失败 3 项：近 252 日置信度 Brier，以及近 20 日上涨比例差、近 60 日最长连续看涨。该方案没有改善原资金流向候选。

另以“订单卖压与资金紧张可能共同影响次日方向”为独立假设，`liquidity_moneyflow_downside_v1/` 将 13 项银行间及交易所资金压力特征和 10 项个股资金流向特征共同加入原来的 18 项输入。仍只用开发期选择既有两个模型和五个门槛，没有增加训练参数搜索或按连涨日期构造条件。开发期选中逻辑回归 70% 门槛，两年准确率分别为 49.59%、55.79%，改判 20 次。

联合输入回归期修正 1 次错误、没有新增错误，准确率为 58.31%；近 252、60、20 日准确率分别为 58.73%、58.33%、60.00%。24 项性能条件均通过，但近 20 日仍有 18 次看涨，近 60 日最长连续看涨仍为 15 次，两项纠偏条件失败。两个实验的完整回放均与开发期前缀一致，未发布。

```powershell
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --moneyflow-context artifacts/evaluation/moneyflow_context_20260918 --calibrate-downside --output artifacts/run/new_calibrated_moneyflow
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --moneyflow-context artifacts/evaluation/moneyflow_context_20260918 --liquidity-context artifacts/evaluation/liquidity_context_20260918_v2 --output artifacts/run/new_liquidity_moneyflow
```

## 价格与订单买卖差的联合特征

`moneyflow_price_context_20260918/` 完全复用已有 1,416 份资金流向响应和个股日线，没有新增网络下载。新增六项特征：上涨股票大单卖出金额占比、下跌股票大单买入金额占比、当日成交额前 10% 股票及其余股票的大单买卖差，以及前两项的五日均值。沿用滞后一日和双覆盖率 98% 的要求，旧十项特征逐字段保持不变。三个真实历史截断点的特征回放一致，资金特征测试共 13 项通过。

`moneyflow_price_downside_v1/` 只按开发期选中逻辑回归 55% 门槛，2023、2024 年准确率为 50.00%、57.02%，改判 100 次。

| 范围 | 原模型准确率 | 候选准确率 | 候选下跌召回率 |
| --- | ---: | ---: | ---: |
| 415 日回归期 | 58.07% | 59.04% | 56.91% |
| 近 252 日 | 58.73% | 57.94% | 51.75% |
| 近 60 日 | 58.33% | 55.00% | 70.37% |
| 近 20 日 | 60.00% | 65.00% | 50.00% |

回归期修正 19 次错误、新增 15 次错误。近 20 日看涨从 18 次降为 13 次，近 60 日最长连续看涨从 15 次缩短为 12 次，两项纠偏条件通过；但近 252 日和近 60 日的准确率、平衡准确率、收益 MAE、置信度 Brier 共八项性能条件失败。该候选未发布。

`recent_moneyflow_price_downside_v1/` 检验三年等权训练可能不能及时适应市场关系变化的假设。仅新增一个训练政策：保留 756 日上限，样本权重按交易日年龄指数衰减，半衰期固定为 126 日，并归一化为均值一以保持正则化尺度。标准化和模型拟合使用相同权重。模型参数、最少 120 行、每五日重训、两个模型及五个门槛均不变；仍只在两个开发年份选择一个组合，沿用原 26 项检查。不根据回归结果调整半衰期。

十个组合均未通过开发期。逻辑回归五个门槛的 2024 年准确率均低于原模型 54.55%；浅层树的 55%、60%、65% 门槛分别有年度准确率或平衡准确率退步，70%、75% 仅改判 3 次和 0 次，未达到至少五次的要求。实验止于开发期，没有读取其回归表现来选择其他半衰期。新增权重及两个真实模型在两种训练政策下的因果性测试通过，连同资金特征与概率校准测试为 32 项通过。

`ensemble_moneyflow_price_downside_v1/` 固定另一个概率集成假设：新增六项价格组合特征与旧资金特征存在相关性，单个模型的系数可能对输入选择敏感。分别训练原十项资金特征版本和价格扩展版本，将两者的下跌概率固定等权平均后再使用原有门槛。逻辑回归和浅层树各构成一个候选，仅在开发期选择一次；两个组成模型均须完成训练才能改判。不按近期具体日期选择组成模型，也不调整平均权重。

开发期选中逻辑回归 55% 门槛，2023、2024 年准确率为 50.00%、57.44%，改判 103 次。只有此组合进入回归。

| 范围 | 原模型准确率 | 集成准确率 | 集成下跌召回率 |
| --- | ---: | ---: | ---: |
| 415 日回归期 | 58.07% | 59.52% | 56.35% |
| 近 252 日 | 58.73% | 58.73% | 49.12% |
| 近 60 日 | 58.33% | 61.67% | 66.67% |
| 近 20 日 | 60.00% | 70.00% | 40.00% |

回归期修正 18 次错误、新增 12 次错误。四个窗口的全部 24 项性能条件通过，近 20 日看涨从 18 次降至 16 次，上涨占比差从 40 个百分点缩小至 30 个百分点；但近 60 日最长连续看涨仍为 15 次，故仅通过 25/26 项，未发布。不能把近 20 日增加的两次命中视为已经解决连续看涨，也不能把反复观察过的回归窗口称为独立验证。

完整回放与开发期前缀逐字段一致。集成后的专用模型测试为 12 项通过，覆盖两种真实模型、两种训练政策、未来结果隔离、缺失训练回退、权重尺度以及不同 DataFrame 行索引的日期对齐。生产四个核心源码与已发布 9 月 17 日 CSV 哈希再次确认未变。

```powershell
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --moneyflow-context artifacts/evaluation/moneyflow_price_context_20260918 --moneyflow-price-features --training-policy recent --output artifacts/run/new_recent_moneyflow_price
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --moneyflow-context artifacts/evaluation/moneyflow_price_context_20260918 --moneyflow-price-features --moneyflow-price-ensemble --output artifacts/run/new_ensemble_moneyflow_price
python -m pytest tests/test_downside_specialist.py tests/test_moneyflow_features.py tests/test_downside_probability_calibration.py -q
```

## 资金流向的滚动相对尺度

开发期原始资金数据表明，大单买卖差比例均值由 2021 年的 -0.0131 变为 2024 年的 -0.0096，标准差由 0.0105 变为 0.0088；上涨股票大单卖出金额占比的标准差则由 0.0569 变为 0.1035。固定训练区间标准化不能区分相同绝对值在不同阶段是否异常，因此新增一个预先固定的特征变换假设。

每项资金特征用其源日期严格之前的 126 个国内交易日计算均值和总体标准差，转换为相对偏离，截断在正负五个标准差内；然后依旧滞后一个交易日作为预测输入。必须有完整的 126 日历史且标准差大于 1e-12，否则标为不可用，不填零、不延用旧值。保留原 18 项非资金输入，原资金十项和价格扩展六项均采用这个变换。仍使用原两个模型、均匀训练权重和五个门槛，仅在开发期选择一次，不同时搜索窗口或截断值。

`normalized_moneyflow_price_downside_v1/` 开发期选中浅层树 55% 门槛，2023、2024 年准确率为 50.41%、56.20%，改判 63 次。415 日、近 252 日、近 60 日、近 20 日的回归准确率分别为 55.66%、57.14%、56.67%、60.00%。虽将最长连续看涨由 15 次缩短为 11 次，回归期修正 17 次错误、新增 27 次错误，仍有 15 项性能条件失败，未发布。资金特征共 15 项测试通过，包含归一化统计排除当前源值、极值截断和缺失或零尺度处理。真实特征在三个历史截断点精确一致；1,397 个信号日中前 126 日因预热不足不可用，首个可用日为 2021-06-24，这些日期保留原预测并继续参与成绩比较。

## 宽基 ETF 份额变化

[Tushare fund_share 文档](https://tushare.pro/document/2?doc_id=207) 提供以万份为单位的历史基金份额，单次最多 2,000 行。本轮固定四只 2020 年前已上市的宽基 ETF：上证 50 的 510050.SH、沪深 300 的 510300.SH、中证 500 的 510500.SH 和创业板的 159915.SZ，不按今天的规模排名回选历史样本。研究假设是这些基金的份额增减可能提供不同于二级市场订单买卖差的信息；份额增减不等于现金流，份额拆分等公司行为也可能影响它。

`etf_context_20260918/` 按基金和年份保存 28 份原始响应及哈希，覆盖 2020-11-13 至 2026-09-14，合计 5,690 行。所有所需交易日都有记录。三个沪市基金各有五条非交易日披露，创业板基金有七条，保留原始数据但不纳入交易日窗口。没有重新获取全部 A 股行情。

每只基金提取份额一日变化比例、五日变化比例及一日变化的 20 日均值，共 12 项输入，全部滞后两个国内交易日。历史响应没有原始送达或修订时间戳，两日滞后属于研究约束，不是历史实际可用性证明。缺失日期不视为份额未变，不前向填充；任意基金窗口不完整，则当日不训练或改判。全部 1,397 个基线信号日特征可用。

`etf_downside_v1/` 将这 12 项份额特征加入原有 18 项价格、广度、海外及置信度输入，沿用两个辅助模型、五个门槛、均匀训练权重和原 26 项验收，不混入资金流向候选或根据回归结果更换 ETF。

十个组合均未通过开发期。逻辑回归 65% 门槛的两年准确率与基线持平，但 2023 年平衡准确率下降；其余逻辑回归门槛至少一年准确率下降。浅层树 55% 两年均退步，60% 仅四次改判，65% 及以上没有改判。未继续进入回归，也未替换模型。

28 份原始响应哈希和全部 1,397 行已保存特征完成复核，三个真实历史截断点与全量计算逐字段相同。ETF 和归一化资金特征共 24 项测试通过，包含交易日滞后、份额单位不变性、非交易日披露排除、缺失传播、未来数据隔离与请求错误去敏。

```powershell
python tools/fetch_etf_context.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_etf_context
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --etf-context artifacts/run/new_etf_context --output artifacts/run/new_etf_downside
```

## 随机森林条件下跌模型

在相同的资金流向与价格输入上，新增一个固定的随机森林候选检验非线性交互与抽样平均的作用。采用 scikit-learn 实现，200 棵树、深度上限三层、叶节点至少 20 个样本、每次分裂考虑一半特征、有放回抽样、随机种子 42。仍使用 756 日窗口中此前已结算的看涨样本，每五日重训，样本权重均匀。输入为原 18 项和未归一化的 16 项资金及价格组合特征，五个改判门槛不变。仅运行这个新增模型并在两个开发年份选择一个门槛，不重新搜索此前模型或随机种子。

`bagged_moneyflow_price_downside_v1/` 开发期选中 60% 门槛，2023、2024 年准确率分别为 49.59%、55.79%，改判十次。新增随机森林也通过真实模型的未来结果隔离和前缀一致性测试；专用模型相关测试为 14 项通过。

完整回放与开发期前缀一致，但回归仍失败三项：415 日收益 RMSE 为 0.0089151366，高于原模型 0.0089144582；近 20 日上涨占比差仍为 0.4，近 60 日最长连续看涨仍为 15 次。误差上升虽小，也没有放宽原门槛。该候选仅通过 23/26 项，未发布。

```powershell
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --moneyflow-context artifacts/evaluation/moneyflow_price_context_20260918 --moneyflow-price-features --model-family bagged --output artifacts/run/new_bagged_moneyflow_price
```

## 期权持仓的期限分组与连续合约比较

先核查随机森林的开发期概率：在原模型看涨且辅助模型已训练的记录中，40%-50% 概率段有 126 行，平均下跌概率 45.36%，实际下跌 45.24%；50%-60% 段有 105 行，分别为 53.88%、50.48%。高于 60% 的十行虽实际下跌八次，样本不足以据此调整门槛。此次没有在已观察回归结果后追加随机森林校准试验。

新的独立假设是总体认沽/认购持仓比混合了不同期限，并且到期和新上市合约会改变样本组成。`option_position_context_20260918_v2/` 完全复用已有 71 个月的期权日线与合约元数据，不产生网络请求；距摘牌日不超过 30 个日历日作为近到期的近似，其余为远期。日期依据每个历史观察日计算，不使用今天的合约名单筛选历史。

新增八项特征：近到期和远期各自的认沽持仓占比、近到期总持仓占比、两个期限的认沽占比差、相邻交易日共同存在合约的认沽和认购持仓增长率、两侧持仓增量差相对于共同合约此前总持仓的比例，以及最后一项的五日均值。到期退出和新上市合约不计入共同合约的持仓增量，同时保留共同合约数量与其覆盖的此前持仓占比供数据核查。

保留原九项期权指标，旧聚合值逐字段精确一致。新特征依旧滞后一日，近到期或远期没有持仓、分母为零或滚动数据不足则标为不可用，不填零或前向填充。1,416 日聚合对应 1,397 个信号日，其中 50 日不可用，这些日仍保留原预测并参与验收。历史元数据和日线没有原始送达或修订时间戳，这个回放不能视为事前记录。

初次构建目录 `option_position_context_20260918/` 因 CSV 自动推断整数和浮点类型的差异而中止，没有模型结果。修复类型比较后在新目录重建，仍要求数值精确相等，不放宽数值容差。

模型实验固定为原 18 项基础输入加九项旧期权指标及八项新指标，使用原逻辑回归与浅层树、均匀样本权重和五个改判门槛，只在 2023、2024 年选出一个组合。六项新增测试通过，覆盖到期及新合约不被误算为共同合约变化、期限分组随历史日期变化、缺失与零分母处理、完整合约集检查以及前缀与未来数据隔离。

`option_position_downside_v1/` 开发期选中逻辑回归 65% 门槛，2023、2024 年准确率为 49.17%、55.79%，改判 45 次。回归期修正 11 次错误、新增十次错误，415 日、近 252 日、近 60 日和近 20 日准确率为 58.31%、59.13%、60.00%、60.00%。但近 20 日收益 RMSE、置信度 Brier 变差，且看涨比例差和最长连续看涨均未改善，四项验收失败，未发布。全部已保存特征和三个真实历史截断点已精确核验。

随后独立验证数据时点：`option_position_close_context_20260918/` 复用原有日线，并仅补取 2026-09-16 一个源日期，使最后一个已结算信号日也具备当日完整期权输入。旧聚合数值保持不变。新实验统一假定晚间 20:00 才生成信号，使用当日完整期权活动和持仓；研究入口拒绝早于 20:00 的发布时间设定，也拒绝缺少当日期权时静默沿用旧值。[官方日线接口说明](https://tushare.pro/document/2?doc_id=159) 没有提供原始送达时刻，因此 20:00 只是研究假设，若通过历史验收仍需实际送达时间和事前预测验证，不自动修改服务发布时间。

模型、特征、训练窗口和五个门槛均沿用期限分组实验，仅改变数据时点并重新按开发期筛选。期权与期限分组共 21 项测试通过，覆盖旧滞后一日行为和当日政策的时钟约束、前缀回放与未来行情隔离。

`option_position_close_downside_v1/` 开发期选中逻辑回归 55% 门槛，2023、2024 年准确率为 50.41%、59.50%，改判 107 次。415 日、近 252 日、近 60 日、近 20 日的回归准确率为 55.18%、57.14%、58.33%、60.00%。近 20 日看涨从 18 次降至十次，近 60 日最长连续看涨从 15 次缩短为三次，但回归期修正 31 次错误、新增 43 次错误，12 项性能条件失败，未发布。最新补取的 694 行原始期权日线已验哈希，全部 1,397 行特征及三个真实前缀精确核验。

下一项探索明确源于已经观察到的取舍：资金流向与当日期权的误差模式不同。固定分别训练原十项资金流向输入的模型与当日期权期限分组模型，各自包含共同的 18 项基础输入；两者下跌概率固定各占一半。模型使用相同的可用样本掩码，任一组输入不可用则不训练或改判。逻辑回归和浅层树各形成一个组合候选，仍只在两个开发年份选择原五个门槛，不按回归日期调权重。该实验是探索性研究，不是未见样本检验，若通过也不构成未来不回退的证明。

`option_moneyflow_ensemble_v1/` 开发期选中逻辑回归 55% 门槛，两年准确率为 50.00%、57.85%，改判 102 次。415 日、近 252 日、近 60 日、近 20 日的回归准确率为 59.28%、60.71%、61.67%、65.00%，回归期修正 24 次错误、新增 19 次错误。通过全部 24 项性能条件，近 20 日看涨从 18 次降为 17 次，但近 60 日最长连续看涨仍为 15 次，只有 25/26 项通过，未发布。

继续检查该组合的开发期原始概率，2023 年资金流向和期权的 Brier 分别为 0.262231、0.269663，2024 年分别为 0.256959、0.247970，两个来源的相对误差发生变化。由此固定一个因果权重学习方案：在此前 252 个信号日内，至少有 60 个已结算、原模型看涨且两侧均已训练的预测，才拟合一个限制在 0 至 1 的资金流向权重；期权权重为一减该权重。用 SciPy 最小化混合概率的 Brier，加上相当于 20 个样本、按两侧概率差平方均值缩放的正则项，将权重收缩至 0.5。样本不足或两个概率序列相同时保留等权，不查看当前或未来结果。

这个权重政策在所有日期和两个模型上相同，没有搜索窗口、正则强度或按具体连涨日期选权重；组合完成后仍沿用原五个门槛和开发期筛选规则。

`adaptive_option_moneyflow_ensemble_v1/` 开发期选中逻辑回归 65% 门槛，两年准确率为 50.00%、55.79%，改判 45 次。415 日、近 252 日、近 60 日、近 20 日回归准确率为 58.31%、58.73%、58.33%、60.00%，修正七次错误、新增六次错误。回归 MAE/RMSE、近 252 日 MAE/Brier 和两项纠偏条件共六项失败，未发布，也未回头选其他门槛。

权重模块四项测试通过，覆盖当前及未来结果隔离、原看跌和未结算记录排除、基础模型未训练时不参与学习、滑动窗口、相同预测保留等权、日期与训练时点拒绝。两个组合的 1,397 行基础模型概率逐字段完全相等，确认新实验只改变组合权重；权重学习日期均早于对应信号日。完整回放与开发期前缀精确一致。加上期权特征，本轮共有 25 项相关测试通过；生产四个核心源码及原已发布 CSV 哈希未变。

```powershell
python tools/build_option_positions.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --options artifacts/evaluation/option_context_20260917 --include-current --output artifacts/run/new_option_positions_close
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --option-context artifacts/run/new_option_positions_close --option-position-features --option-lag-sessions 0 --publication-hour 20 --moneyflow-context artifacts/evaluation/moneyflow_context_20260918 --option-moneyflow-ensemble --output artifacts/run/new_option_moneyflow_ensemble
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --option-context artifacts/run/new_option_positions_close --option-position-features --option-lag-sessions 0 --publication-hour 20 --moneyflow-context artifacts/evaluation/moneyflow_context_20260918 --option-moneyflow-ensemble --adaptive-context-ensemble --output artifacts/run/new_adaptive_option_moneyflow
python -m pytest tests/test_option_features.py tests/test_option_position_features.py tests/test_downside_ensemble.py -q
```

## 沪市历史市值加权信号

新的研究假设是市场口径不一致：原新增资金指标主要聚合全部沪深股票，广度多按股票数量统计；即使有上海市场相对指标，也没有显式表示沪市大市值股票的权重。这里独立检验历史市值加权的沪市广度与资金流向，不因已知 15 日连续区间逐日改变规则。

[Tushare daily_basic 官方文档](https://tushare.pro/document/2?doc_id=32) 提供交易日总市值 `total_mv`，单位万元，说明更新时段为 15:00 至 17:00，单次最多 6,000 行。只新增这一历史字段及股票代码、交易日期；原个股日线和资金流向均复用已冻结缓存并逐文件校验哈希。全日期分页获取后，仅使用当日活跃、代码为 `6xxxxx.SH` 的沪市 A 股，按当时总市值加权或分组，不用今天的股票名单回选历史。它是沪市市场结构代理指标，不是上证指数成分、纳入规则与除数的精确重建。

固定七个日度量：市值加权上涨占比、市值加权收益、加权与等权上涨占比之差、市值加权大单净额比例、市值加权大单卖出占比、按当日市值排名前 10% 股票和其余股票各自的大单净额比例；再加资金比例与收益的五日均值，以及上涨占比一日变化，共十项特征。大单净额比例的分母沿用四档买卖金额之和。新特征全部滞后一个国内交易日，官方更新时间仅支持研究约束，不能证明历史首次实际送达时间。

任何活跃沪市股票的市值缺失或不为正，整日指标不可用，避免不知道缺失股票权重时用股票数量近似覆盖率。资金流向匹配需同时覆盖至少 98% 股票数、成交额和总市值；不足时不训练或改判，不填零或延用旧值。缺失交易日、重复股票及错误日期直接拒绝。

本轮模型事先固定为原 18 项基础输入加上述十项特征，沿用普通逻辑回归与浅层树、756 日训练窗口、至少 120 个此前已结算的原看涨样本、每五日重训、均匀样本权重和 55% 至 75% 五个既定门槛。仍只在 2023、2024 两个年份按原规则选出一个组合；未通过开发期则不进入回归，通过后仅该组合接受原 26 项检查。不会在看过回归结果后换市值分组、训练窗或门槛。

市值特征的 13 项测试通过，含权重与单位不变性、跨市场排除、未知市值拒绝、缺失大权重股票不能被高股票数覆盖掩盖、滞后及缺失传播、真实分页调用和请求错误去敏。加上既有专用模型测试，共 27 项通过。已有冻结影子候选 `1b626719...d2d6f1f` 的源码、参数和输入合同仍然通过校验。

市值数据已经完整获取：20201117 至 20260915 共 1,416 个源交易日、7,175,541 行市值记录。独立校验全部 4,248 份市值、行情和资金流向原始响应哈希，并精确重算聚合值及 1,397 行特征，无不可用信号日；20221230、20241231、20260916 三个截断点分别复现 498、982、1,397 行特征。证据在 `capital_context_20260918/verification.json`。历史首次送达时间仍然未知。

`capital_downside_v1/` 开发期选中逻辑回归 65% 门槛，2023、2024 年准确率为 51.24%、56.20%，改判 39 次。回归期修正 15 次错误、新增 14 次错误，415 日准确率从 58.07% 升至 58.31%，但近 252 日从 58.73% 降至 57.94%；近 60 日和近 20 日仍为 58.33%、60.00%。近 20 日依然看涨 18 次，近 60 日最长连续看涨仍为 15 次。七项验收失败，仅 19/26 项通过，未发布。

### 当日价格与滞后资金流向

下一项独立时点假设依据接口文档：`daily_basic` 在 15:00 至 17:00 更新，价格和市值加权广度可以在 18:00 之后使用当日完整数据；`moneyflow` 文档没有明确历史送达时间，因此仍保留其一交易日滞后。研究将十项特征中的五项价格指标改为当日值：加权上涨占比、加权收益、加权与等权上涨占比之差、加权收益五日均值和上涨占比一日变化；其余五项资金指标完全沿用已审计的滞后一日值。价格指标独立使用全部当日活跃沪市 A 股，不依赖资金流向是否匹配；任何活跃股票缺少正市值则价格特征不可用，不填充。

只补取最后一个已结算信号日 20260916 的市值，其余市值和个股行情复用已验哈希的历史响应。所有价格输入必须是信号日当天完整数据，入口拒绝 18:00 前的研究发布时间；三个截断点检查时保留当天价格、删除未来价格。原 18 项输入、两个模型、756 日训练窗、均匀权重、每五日重训、至少 120 个已结算看涨样本及五个门槛均保持不变；开发期仅选择一次，再按原 26 项条件验收。历史时间假设依然不能代替实际收到数据的时间记录，本轮结果仍是已观察数据上的探索。

`capital_close_context_20260918/` 构建并审计完成，共 1,417 个源日期，新增请求仅 20260916 的市值。1,397 行特征均可用，三个截断点逐字段精确复现。31 项市值及时点、专用模型测试通过，原固定影子版本继续通过源码和依赖校验。

`capital_close_downside_v1/` 开发期选中逻辑回归 60% 门槛，2023、2024 年准确率为 50.00%、59.92%，改判 67 次。415 日、近 252 日、近 60 日、近 20 日准确率分别为 58.31%、57.94%、60.00%、65.00%；近 20 日看涨降至 17 次，近 60 日最长连续看涨缩短至 11 次。回归期修正 26 次错误、新增 25 次错误，但六项性能验收失败，仅通过 20/26 项，因此未发布。

基于新价格信号能改变连续区间但误改判偏多的结果，下一轮固定将当日市值候选与既有资金及价格组合的下跌概率等权平均。资金组合内部仍是十项资金特征与十六项资金加价格特征各占一半，因此三组概率最终权重为 50%、25%、25%；不搜索权重。三个模型均包含原 18 项基础输入，采用共同可用日期和原训练政策；逻辑回归与浅层树分别形成候选，只在两个开发年份按原规则选择一个模型及门槛，再接受原 26 项回归条件。此假设是在已观察结果上提出，仍明确属于探索，不作为新的未见测试集。

`capital_moneyflow_downside_v1/` 开发期选中逻辑回归 55% 门槛，2023、2024 年准确率为 50.83%、58.26%，改判 103 次。415 日、近 252 日、近 60 日、近 20 日准确率分别为 59.76%、58.73%、61.67%、70.00%，回归期修正 30 次错误、新增 23 次错误，近 20 日看涨降为 16 次。但近 252 日收益 RMSE 为 0.0090106427，高于基线 0.0090031322；近 60 日最长连续看涨仍为 15 次。仅 24/26 项通过，未发布，也没有按回归结果改权重或换门槛。

三轮实验均保存输入、源码、开发期选择和逐项验收结果。两个当日价格实验的完整模型回放与开发期前缀逐字段一致。该轮结束时生产四个核心文件继续匹配原始冻结哈希，公开 CSV 的 SHA256 为 `49d889325b8a44a5de4e3aa9ab44aa31d787b1d54a207be8a6eea2652abe0d86`。该轮研究发现当日市值价格特征可以缩短连续看涨，但尚未满足全部不回退条件。

复现时使用新的输出目录，已完成的目录拒绝覆盖：

```powershell
python tools/build_capital_close_context.py --capital artifacts/evaluation/capital_context_20260918 --breadth artifacts/evaluation/market_breadth_close_20260917 --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_capital_close
python tools/build_capital_close_context.py --capital artifacts/evaluation/capital_context_20260918 --breadth artifacts/evaluation/market_breadth_close_20260917 --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_capital_close --verify
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --capital-context artifacts/run/new_capital_close --capital-close-prices --output artifacts/run/new_capital_close_model
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --capital-context artifacts/run/new_capital_close --capital-close-prices --moneyflow-context artifacts/evaluation/moneyflow_price_context_20260918 --moneyflow-price-features --moneyflow-price-ensemble --capital-moneyflow-ensemble --output artifacts/run/new_capital_moneyflow_model
python -m pytest tests/test_capital_close_features.py tests/test_capital_features.py tests/test_downside_specialist.py -q --basetemp artifacts/run/new_capital_tests
```

### 当日市值模型的概率刻度

只检查当日市值逻辑回归在 2023、2024 开发期的原看涨且已训练记录，固定按 0.4、0.5、0.6、0.7、0.8 分档。2023 年 60%-70% 档有 20 次，平均预测下跌概率 63.33%，实际下跌比例 50.00%；2024 年同档有 24 次，平均预测 64.43%，实际下跌 75.00%。50%-60% 档分别为 49、35 次，平均预测 55.24%、54.36%，实际下跌 48.98%、40.00%。开发期原始下跌概率 Brier 为 0.260220、0.246062。这些小样本差异不能证明稳定校准关系，但为检验随时间变化的概率刻度提供依据。

下一轮仅给当日市值候选应用已有 `calibrate_downside_probabilities`：在此前 252 个信号日内，只用原模型看涨、辅助模型已训练且结果已结算的预测；至少 60 条且两类结果均出现才拟合递增 isotonic 校准，否则保留原概率。每个时点排除当前和未来结果，模型窗口、特征、训练政策和两个模型均保持不变；仍在 2023、2024 年选择五个既定门槛中的一个，再仅对选中组合进行原 26 项验收。不联合搜索校准窗、分档、方法或改判门槛。

`calibrated_capital_close_downside_v1/` 开发期选中逻辑回归 75% 门槛，2023、2024 年准确率为 49.17%、54.96%，改判 11 次。回归期修正三次错误、新增一次错误，415 日、近 252 日、近 60 日、近 20 日准确率分别为 58.55%、59.13%、60.00%、60.00%。但近 20 日 MAE/RMSE 回退，两项偏向条件均未改善，最长连续看涨仍为 15 次；通过 22/26 项，未发布。八项校准测试通过，没有按后续结果换校准方法或重新挑门槛。

### 按原方向分别训练的双向纠错

当前市值专用模型只训练原看涨样本并允许改判为看跌。开发期原看跌记录也存在较多错误：2023 年 126 次中错误 63 次，2024 年 90 次中错误 41 次。因此单向纠错缺少修正错误看跌的能力，本轮检验在相同信息下分别学习两种原方向的错误概率。

冻结方案为当日市值版本相同的 28 项输入，两个独立分支分别训练此前原看涨与原看跌且已结算的样本，目标均是原方向是否错误。沿用 756 日窗口、至少 120 个对应方向样本、每五日重训及均匀样本权重。零收益仍属于看跌，所以原看跌且零收益不是错误。逻辑回归与浅层树各构成一个双向候选，两个分支共用同一改判门槛；仍仅比较既定五个门槛，两个开发年份按原准确率与平衡准确率规则选择一次，再接受原 26 项条件。不更改原预测账本、性能标准或已冻结影子候选。

`directional_capital_close_v1/` 开发期选中逻辑回归 55% 门槛，两年准确率为 51.24%、57.85%，改判 173 次。415 日、近 252 日、近 60 日、近 20 日准确率分别为 55.66%、55.16%、58.33%、65.00%，近 20 日看涨 15 次，最长连续看涨缩短至 11 次。但回归期修正 54 次错误、新增 64 次错误，其中原看跌分支改判 45 次，只改对 18 次；原看涨分支改判 73 次，改对 36 次。15 项性能验收失败，仅 11/26 项通过，未发布，也未重新选择门槛。

五项新增测试通过，包含真实逻辑回归与浅层树的双向改判、两类样本训练隔离、零收益标签、未结算样本排除、缺失特征不得改判、阈值与训练时点拒绝、当前及未来数据隔离及完整历史与前缀逐字段相等。真实开发期原看涨分支的概率、训练样本数和最后训练日期与 `capital_close_downside_v1` 精确一致；完整模型回放也与开发期的全部概率和训练诊断精确一致。这个实验没有解决性能取舍。

```powershell
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --capital-context artifacts/evaluation/capital_close_context_20260918 --capital-close-prices --bidirectional --output artifacts/run/new_directional_capital
python -m pytest tests/test_directional_specialist.py -q --basetemp artifacts/run/new_directional_tests
```

### 按已结算收益幅度加权训练

此前专用分类模型以每次方向错误相同的权重训练，而验收还包含收益 MAE/RMSE。这个目标差异独立于具体连续看涨日期：同样一次错误，行情幅度较大时会产生更大的收益预测损失。本轮固定检验使用历史绝对收益作为样本权重的模型，不搜索权重指数或截断分位数。

输入仅使用当日市值版本原 28 项特征，两个既定模型、756 日训练窗、至少 120 条此前已结算的原看涨样本、每五日重训均保持不变。每次拟合只用当时训练窗内的已结算结果，权重为绝对收益除以这些训练样本的平均绝对收益，均值为一；零收益没有幅度损失，不参与拟合。标准化与分类器使用相同权重，记录有效样本量和单样本最大权重占比供核查，不用回归期设置裁剪。

加权分类分数表示条件下跌收益在总绝对收益中的占比估计，不能直接解释为下跌发生概率；原有置信度校准仍根据过去真实命中重新估计，报告明确记录这一语义。仍仅在两个开发年份选择既定两个模型和五个门槛，按准确率和平衡准确率原规则选一次，再检查原 26 项条件。历史数据已被查看，本轮是探索，不构成未见样本性能证明。

`magnitude_capital_close_v1/` 开发期选中浅层 XGBoost 55% 门槛，两年准确率为 53.31%、56.61%，改判 57 次。415 日、近 252 日、近 60 日、近 20 日准确率为 58.80%、57.14%、58.33%、70.00%；近 20 日看涨为十次，与实际上涨次数相同，近 60 日最长连续看涨由 15 次缩至五次。但回归期修正 28 次错误、新增 25 次错误，近 252 日准确率等九项性能条件失败，仅 17/26 项通过，未发布。

四项测试通过，覆盖真实逻辑回归和浅层树的未来隔离及历史前缀、原看跌不改判、收益量纲不变性、频繁小跌与少量大涨的收益占比、零收益和未知结果排除。完整回放与开发期全部模型分数和训练诊断精确一致。回归期模型已训练的 254 个看涨日，有效训练样本量最少 164.45、中位数 179.67，单一样本最大权重占比 3.00%；未因失败结果追加权重裁剪或改选其他门槛。

```powershell
python tools/evaluate_downside_specialist.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --breadth artifacts/evaluation/market_breadth_close_20260917 --global-context artifacts/evaluation/global_risk_data_20260917_complete --capital-context artifacts/evaluation/capital_close_context_20260918 --capital-close-prices --magnitude-weighted --output artifacts/run/new_magnitude_capital
python -m pytest tests/test_magnitude_downside.py -q --basetemp artifacts/run/new_magnitude_tests
```

### 规则阈值状态的持续长度

另一个独立候选 `duration_conditioned_probability` 检验某项二值阈值状态持续时间是否改变其条件可靠性。保留原 120 日准确率排名、前五名权重及状态否决和失败保护，将每条规则的状态持续时间固定分为第一日、第二至五日、第六至二十日、二十一日以上四档；基础状态概率采用 Beta(1,1)，各时长档用十个基础状态伪样本收缩，方向门槛固定 0.5。不搜索档位、收缩强度或门槛，不以预测连续次数强制反向。

`duration_rules_v1/` 的开发期结果：2023 年准确率由 48.35% 升至 50.41%，平衡准确率由 48.28% 升至 50.63%；2024 年准确率由 54.55% 降至 51.24%，平衡准确率由 54.23% 降至 50.92%。改判 145 次，因未通过第二个开发年份而停止，没有运行 2025 年之后的回归，也没有发布。

六项测试通过。独立截断至 20240628，删除其后的行情和目标列，端点结果为空，857 行方向、收益、置信度及收益缩放与完整开发回放精确一致，证据在 `duration_rules_v1/prefix_verification.json`。生产核心文件与公开 CSV 哈希保持原值。

```powershell
python tools/evaluate_duration_rules.py --input artifacts/evaluation/direction_debias_v1/features.csv --baseline artifacts/evaluation/direction_guard_v2/comparison/baseline_predictions.csv --output artifacts/run/new_duration_rules
python -m pytest tests/test_duration_rule_candidates.py -q --basetemp artifacts/run/new_duration_tests
```

## 固定资金流向组合的影子留痕

将 `ensemble_moneyflow_price_downside_v1` 已选定的逻辑回归 55% 门槛接入独立研究入口，仍为原十项资金流向模型与十六项资金及价格模型的下跌概率等权组合。原 26 项验收维持 25 项通过，最长连续看涨仍为 15 次。没有重新选择门槛，没有提供生产晋升入口，也没有在控制面板增加纠偏展示。

冻结版本保存于 `artifacts/evaluation/moneyflow_price_shadow_v1/`，release 为 `1b6267194f359228e869fdc9bf45866baa60885e4e6578b541c41bfb3d2d6f1f`。包含候选参数、种子输入、原失败报告、相关源码 ZIP 及 NumPy、pandas、scikit-learn、SciPy、XGBoost、PyTorch 版本；更改这些内容会拒绝运行，必须显式冻结新版本。冻结时逐字段复现原候选的 1,397 条预测。

`tools/run_downside_shadow.py` 使用已发布的不可变行情快照，先验证原模型回放与公开账本，再计算候选。候选共用现有 `ModelRelease`、`ShadowRun`、`PredictionLedger` 和 `OutcomeResolution`，独立版本写入，后续只追加结算，不切换公开 `Publication`。修正了新影子模型可能继承另一模型相同预测的旧生成时间的问题；正式兼容发布保留原有继承逻辑。面板原 BiLSTM 查询只选择 BiLSTM 记录，避免把新候选误显示为 BiLSTM。

事前记录必须有完整交易日历，实际生成时间在信号日上海时间 18:00 之后、下一交易日 09:30 之前；归档结束时再次检查是否已开盘。开盘后默认拒绝新写入，只有明确传入 `--allow-backfill` 才允许作为回放保存。事前统计只比较候选与控制模型均在开盘前留痕、且具有相同结算交易日和实际收益的记录。

2026-09-18 的真实联调只补取以下数据，未重下整个历史：20260917 个股日线 5,553 行、20260916 个股日线 5,550 行、20260916 资金流向 5,550 行，以及 20260916 的 SPX 和 IXIC 各一行。特征计算仍限定沪深市场，排除北交所代码，资金流向滞后一日。新增原始响应带实际收取时间和哈希；既有种子历史没有被伪称为历史时点快照。

真实运行 `cc5247c3-a341-475d-a82a-3071b0cf4b94` 基于正式 20260917 行情快照，写入 1,398 条候选预测及 1,397 条结算。61 行 CSV 位于 `service_data/shadow_archives/2026/09/17/cc5247c3-a341-475d-a82a-3071b0cf4b94/results.csv`。写入时间为 2026-09-18 11:57:30（上海），所以包括目标 20260918 的待结算预测在内，全部标记 `backfill`；事前配对样本数为零。重复运行返回同一 run，不重新取数或写入预测。正式活动 publication ID 及 CSV 哈希 `49d889325b8a44a5de4e3aa9ab44aa31d787b1d54a207be8a6eea2652abe0d86` 未变。

`tools/verify_downside_shadow_replay.py` 独立截断到 20250102、20260902，删除之后的行情、外部数据及预先生成的目标列，再从头运行正式基线和固定候选。983 行和 1,387 行前缀的方向、收益、收盘价、置信度、收益缩放和全部下跌概率及训练诊断逐字段一致，端点实际收益为空。证据保存于冻结目录的 `prefix_verification/`。这是历史因果性核验，不是事前成绩。

44 项相关测试通过，包含真实数据集成、生成时间隔离、开盘时间拒绝、账本不可更新、重复运行、跨日结算、无事前证据的回放排除、冻结源码及种子篡改拒绝、增量请求日期和公开发布不变。测试使用单独 `--basetemp`，避免仓库默认临时目录被并行 pytest 进程同时清理。

```powershell
# 正式行情更新后，对同一个固定版本运行；这里不会主动更新正式行情。
python tools/run_downside_shadow.py run --bundle artifacts/evaluation/moneyflow_price_shadow_v1 --fetch
python -m pytest tests/test_downside_shadow.py tests/test_prediction_service.py tests/test_service_reliability.py tests/test_shadow_live_replay.py -q --basetemp artifacts/run/downside_shadow_tests
```

手动入口保留。新增独立 `tools/watch_downside_shadow.py` 后台进程，默认每 60 秒检查活动正式快照，使用同一个已冻结 bundle。未更新快照或已有对应记录时只核验归档和统计，不重新取数、训练或写预测；新快照只在信号日 18:00 之后、目标交易日 09:30 之前执行原研究命令，始终禁用补录。原 runner 在归档完成时再次检查真实时间，防止长任务跨越开盘后仍记为事前预测。

该进程不刷新正式行情，不启动或停止已有 Web 服务，不更改生产发布，也没有新增面板入口。正式行情需要由现有服务的定时或手动刷新产生，机器和后台进程需要保持运行。网络或计算失败每十分钟重试，同一快照最多六次，次数持久保存，重启不会重置；新快照重新计数。与原 CLI 共用 `downside_shadow.lock` 防止并行写入，另有进程级 `downside_watcher.lock` 防止重复启动。进程状态写入 `service_data/downside_watcher.json`，错误只保留类型，不输出可能含凭据的异常正文。`status` 同时探测实际进程锁和心跳年龄，旧状态文件不代表进程存活。

它没有自动晋升模型的能力，资金流向候选仍失败一项历史纠偏条件。自动留痕用于积累真实事前记录，不代表看涨偏向已解决。

九项后台记录器测试通过，覆盖真实临时账本的首次记录、重复运行、下一快照自动结算、发布不变、持久重试上限、共享 CLI 锁、错过开盘拒绝、冻结版本损坏、归档损坏和错误去敏。既有八项影子账本测试同时通过。真实 `once` 检查识别原 run `cc5247c3-a341-475d-a82a-3071b0cf4b94`，没有新取数或新写预测，事前配对仍为零。2026-09-18 中午已实测启动、进程锁存活检查、请求停止、进程退出及重启；后台记录器已恢复运行，未设置开机自启。正式服务配置为 18:15 刷新，但候选只跟随实际已发布快照，不把配置时间当成发布成功证据。

2026-09-18 13:05 再检查时，旧 PID 39924 已不存在，`status` 正确报告进程锁未占用；最后心跳是 13:03:34，标准错误为空，没有停止请求文件，因此退出原因未确定，没有把旧状态解释成持续运行。13:08:14 恢复为 PID 25804，后续实际进程查询、锁和更新的心跳均验证存活。独立进程尚无系统级重启保障，运行状态需要按 `status` 和实际进程检查，不能承诺关闭电脑或进程退出后仍采集。现有事前配对样本仍为零。

13:33 进一步核查发现 `service_data/service.lock` 未被持有，当前正式服务没有运行；8002/8003 是旧进程，8005 使用演示快照，不能据此证明当前 18:15 调度有运行保障。只读核验当前 runtime release、配置及源码哈希均匹配活动版本 `956dd395...6522e9` 后，13:42:40 用隐藏后台进程启动当前 `run_service.py --port 8010`，PID 19644，日志单独保存为 `service_resume_20260918_134240.*.log`。已验证监听端口、服务锁、healthz/status API 和 CSV API；公开 CSV 仍为 61 行及原哈希，活动 publication 未变，未手动刷新或晋升。尚未到当天定时执行时间，启动成功不是当天刷新成功的证据。

新增只读工具 `tools/audit_downside_prospective.py`，从活动 publication、`prediction_ledger` 和不可变 `outcome_resolutions` 读取数据，按目标交易日 09:30 上海时间重新判定是否为开盘前生成。它列出候选/控制总行数、开盘前行数、已结算行数、共同信号日、单方日期、目标收益不一致日期，以及共同样本的近 20/60/252 日窗口；任何窗口样本不足时明确标为 `sufficient: false`，不把历史回放门槛转化为事前成绩。输出文件采用独占创建，冻结 bundle 先经过 release 校验，整个过程不写数据库。

2026-09-18 14:08（上海）首次真实审计输出为 `moneyflow_price_shadow_v1/prospective_audit_20260918.json`：候选账本 1,398 行全部是补录，候选开盘前行数、结算数和配对数均为 0；控制 release 当前账本 63 行，其中 20260917 有 1 行开盘前预测但尚未结算。因此当前没有任何可用于候选与控制比较的事前样本，`promotion_allowed` 保持 false。新增 4 项审计测试通过。

上一轮会话中断后，13:52 的正式服务 PID 19644 和 watcher PID 25804 均已退出，两个标准错误日志为空；旧状态文件的心跳因此被判定为过期。14:16:36 重新启动正式服务 PID 8848，14:16:38 启动 watcher PID 32928，均使用新的独立日志。实际端口、服务锁、8010 healthz、watcher 进程锁和心跳已重新核验，14:23 仍存活。旧的 20260917 快照未被重复刷新或晋升。

```powershell
python tools/watch_downside_shadow.py run --bundle artifacts/evaluation/moneyflow_price_shadow_v1
python tools/watch_downside_shadow.py status
python tools/watch_downside_shadow.py stop
```

## 强制方向保护的撤回

2026-09-18 至 19 日曾错误地发布 `bullish_streak_guard`：按连续看涨次数和滚动占比把部分预测改为 -0.01%，并把置信度变成其补数。这只改变输出分布，没有新增市场证据。规则、阈值及执行顺序是在查看回归成绩后选择，报告中的“fixed product invariant; no parameter search”与实际过程不符。2025-01-01 生效日是在发现开发期回退后加入，以此让开发期结果保持不变，不能算通过开发期验证。

原验收还使用 `artifacts/baseline/default` 的旧版预测，而生产已经使用固定因果幅度/置信度校准。2026-09-19 审计重新使用实际原模型，数值检查仍可通过 26/26；这不消除样本被反复用于选规则的问题，也不能把强制反向解释成模型能力提升。若正确性 y 和置信度 q 同时取补数，Brier 误差有 `(1-q-(1-y))²=(q-y)²`，其不变本身不提供校准改善证据。

按用户明确要求，已删除服务接入和额外计算前缀、规则模块、专用评估入口及绕过兼容检查的发布入口，恢复原 `956dd395...6522e9` 模型并使用最新 20260918 行情。旧预测、结算和归档保留用于审计；撤回源码存入独立 ZIP，v1–v4 报告标注 withdrawn，不再作为验收成功依据。

## 当前证据的限制

对原模型与始终看涨进行同日配对比较，回归期原模型命中 241/415 次，始终看涨为 234/415 次；两者的不同判断中原模型赢 84 次、输 77 次，双侧精确二项检验的 p 值为 0.6364。近 60 日分别命中 35 次和 33 次，配对赢 15 次、输 13 次，p 值为 0.8506。近 20 日配对仅有 2 次不同判断，p 值为 0.5。这些简单检验没有处理交易日的序列相关性，也不是未来预测保证；未显著不代表两种模型等价，亦不能用来降低候选非回退门槛。

当前基线的有限样本优势不足以支持稳定预测优势的结论，资金流向等研究候选也不能直接替代正式版本。恢复原算法是撤销不合格改动，并非纠偏完成；原目标仍是基于市场信号改善下跌识别，同时保持预测性能。

## 后续模型改进的验证条件

1. 已完成的去重、平衡评分、概率投票和误差模型实验均须保留失败记录。后续候选需要新的建模依据，不应为缩短这一次连续看涨而反复调参。
2. 固定候选和评估口径后，采用时间顺序滚动训练，在未参与选规则的后续数据上比较，逐日只使用当时已结算的信息。已有 2025 至 2026 年的对照已被查看，不能再宣称它是全新的最终测试集。
3. 同时报告两类召回率、与始终看涨基准的配对差异、置信度校准、各月表现和实时留痕成绩。连续长度和切换率只是诊断指标，不以追求更频繁切换作为优化目标。
4. 如增加“信号较弱”或观望状态，须独立评估覆盖率和条件命中率，并为现有二分类 CSV 消费者设计版本兼容方案。不能把观望当作已正确预测。
5. 不得用连续次数上限、方向配额或任意生效日期覆盖模型输出；也不得把开发期关闭、回归期启用当作“各期不回退”。改动必须在开发与验证时使用相同算法，生产比较以恢复的正式模型为基线。

尚无经有效独立验证的替代算法同时满足纠偏和性能不回退。此前完成声明已撤回。

## 2026-09-19：多日趋势参与度

恢复原模型后，重新检验来自实际个股行情的多日趋势广度，区别于此前的单日上涨家数。复用 1,417 日的 7,214,623 条已校验原始记录，计算连续 5/20/60 日收益为正的股票占比、转弱比例等八个信号；缺失与停牌不填补，股票与成交额覆盖率均需达到 90%。本轮没有重新下载全市场历史。

只声明一个逻辑回归候选，沿用已有 756 日历史、120 条最少样本、每 5 日重训与 `C=0.1`，固定 0.60 下跌概率门槛；加入对照以区分新增特征的贡献。开发与回归拟用相同算法，不包含预测次数、方向配额或按年份启停。2023 年方向准确率由 48.35% 升至 50.00%，2024 年由 54.55% 降至 52.89%；2024 年相同日期对照的 Brier 为 0.25459，加入趋势特征后为 0.26367，概率误差也恶化。因此止于开发期，没有再用 2025 年以后数据挑选门槛，生产保持恢复后的原模型。

十项特征测试通过；截断至 2024-06-28 且屏蔽端点结果后，857 行特征和模型概率精确复现。实验输入、训练前合同、消融对照、逐日预测及失败报告见 [trend_breadth_downside_v1](../artifacts/evaluation/trend_breadth_downside_v1/README.md)。这一结果排除了一个具体候选，目标仍保持未完成。

## 多日趋势的内层时间验证

前一轮开发数据表明部分新增输入与已有输入高度相关（最高约 0.936），训练误差略降而后续概率误差变差。下一轮将参数选择放进每次重训的历史数据内：三个连续 40 条已结算原看涨样本的验证块、每折至少 120 条更早训练，分别拟合标准化；依据历史验证 log loss 选择 C 为 0.001/0.01/0.1，以及是否采用整组新增趋势输入，再用全部既往样本拟合。全过程按同一算法运行，没有年份豁免、方向次数条件或外层门槛搜索；0.60 改判门槛保持不变。合同明确记载每次重训的六种配置和全部内层分数。

2023/2024 年开发期准确率分别为 48.76%/55.37%，两年准确率与平衡准确率均通过筛选。后续 415 日、最近 252/60/20 日准确率分别为 58.31%/59.13%/60.00%/60.00%；回归期仅修正一条错误、无新增方向错误。近 20 日 MAE/RMSE 仍有小幅回退，看涨仍 18/20，近 60 日最长看涨仍 15，所以只通过 22/26 项，未发布。151 次重训有 118 次选择原特征，33 次使用扩展特征；2024 年扩展候选概率误差仍差于仅用原特征的内层选择对照，不能声称新信号已提供稳定优势。

六项真实分类器测试通过；截断至 2024-06-28 并屏蔽端点结果后，857 行特征及概率精确复现，完整回归的开发前缀也精确复现。见 [nested_trend_downside_v1](../artifacts/evaluation/nested_trend_downside_v1/README.md)。本轮没有改变生产或降低门槛，整体纠偏目标尚未完成。

## 离岸人民币报价信号

补取 Tushare `USDCNH.FXCM` 的 1,874 条报价，构建 1/5/20 报价日收益、波动率、范围与买卖报价差六项特征。按官方 GMT 日期约定，源报价日期必须严格早于上海信号日。美元指数篮子虽有基本信息，近期日线为空，未作为输入。历史开盘价有 41 条超出高低价的异常，本轮预定特征不使用开盘价；保留原始数据和异常清单，实际使用字段严格校验，没有修补报价。有效上下文在 `fx_context_20260919_v2/`，第一次被开盘字段校验中止的获取记录不当作完整输入。

沿用已经声明的内层时间验证、三种 C、整组特征准入以及 0.60 决策门槛，只测试一个候选。2023 年准确率与原模型同为 48.35%，但平衡准确率从 48.28% 降至 48.14%；2024 年准确率从 54.55% 升至 54.96%，平衡准确率从 54.23% 升至 54.67%。2024 年外汇扩展的概率 Brier 为 0.24990，优于相同训练协议但不含外汇的对照 0.25243；仍不能据此豁免第一年的失败。实验止于开发期，未用后续回归挑选参数、未发布。

三十项相关测试通过，真实数据 857 行截断特征/概率精确复现。输入、报价时点约定、内层搜索、逐日预测与失败结果见 [nested_fx_downside_v1](../artifacts/evaluation/nested_fx_downside_v1/README.md)。正式模型与公开 CSV 保持不变，任务仍未完成。

## 原预测与辅助模型的内层配对选择

进一步检验预测决策的选择目标：之前辅助模型按 log loss 选择，但 2023 年实际纠错 5 次、误改也 5 次。新的内层学习将原预测作为可选项，要求辅助模型在三段此前验证中各自准确率与平衡准确率不下降，合计纠错多于误改，再按 log loss 选参数；不合格时保留原预测。特征、C 的候选范围、0.60 门槛、训练窗和重训频率保持不变，没有预测次数配额或按年份豁免。

开发期 2023 年准确率持平，2024 年从 54.55% 提高至 55.79%，共改判 9 次，因而进入原回归比较。但 415 日回归准确率从 58.07% 降至 57.83%，纠错 4 次、误改 5 次；近 20 日依然看涨 18/20，最长看涨 15 日。只通过 20/26 项，未发布。这也说明历史验证时的“不回退”条件不能替代之后的实际验收。

十一项内层模型测试通过，真实 857 行未知端点前缀精确一致。详见 [paired_nested_fx_downside_v1](../artifacts/evaluation/paired_nested_fx_downside_v1/README.md)。已向用户确认最晚交付时点，以判断能否研究开盘前才完成的隔夜信号；在答复前仍遵守现有晚间预测的可用信息边界。

## 在历史验证中学习决策门槛

在不变更市场信息时点的前提下，进一步把决策门槛放入内层时间顺序验证。固定门槛候选为 0.50/0.55/0.60/0.65/0.70/0.75，与既有 C 和可选外汇特征联合选择；候选每次重训 36 种配置，对照 18 种配置。每个历史验证块仍要求方向准确率和平衡准确率不低于原模型，再按合并的方向指标增量选择。外层没有参数搜索，模型没有预测连续次数或看涨配额输入，未放宽原开发期筛选。

结果更差：2023 年准确率由 48.35% 降至 45.04%，2024 年由 54.55% 降至 53.72%，两年平衡准确率也下降。开发期改判 62 次，2023 年下跌召回提高却伴随上涨召回从 46.15% 降至 29.91%。因此止于开发期，没有再用回归期挑选门槛，没有发布。详见 [learned_threshold_fx_downside_v1](../artifacts/evaluation/learned_threshold_fx_downside_v1/README.md)。

十五项内层学习及每行门槛预测测试通过，真实 857 行未知端点前缀特征、概率和门槛诊断精确复现。这说明实现保持因果，但不代表新规则具有预测增益；纠偏目标仍未完成。

## 共享两种原方向的错误模型

进一步以全部已结算的原看涨和原看跌样本学习“原预测是否错误”，加入原方向及特征乘以原方向的交互项，区别于之前分别训练两个方向的专用模型。仍采用原内层时间验证、C 的三个选项与原预测对照，本轮二分类门槛固定为 0.50，不进行门槛搜索。原市场输入及方向交互为 37 项，加外汇及对应交互为 49 项。

开发期 2023 年准确率由 48.35% 升至 48.76%，2024 年由 54.55% 降至 53.72%，第二年平衡准确率也从 54.23% 降至 53.59%。共改判 127 次，未通过开发期，未进入回归或发布。21 项相关学习模型测试通过，真实 857 行未知端点前缀精确复现，原单向模式与原冻结源码的逐行预测诊断亦精确一致。见 [pooled_fx_error_v1](../artifacts/evaluation/pooled_fx_error_v1/README.md)。

下一数据假设是晚间已完成的指数分钟信息。查阅 Tushare 官方 `idx_mins` 文档并以 `000001.SH` 单日 30 分钟查询探测，当前账号被接口以无访问权限拒绝，尚未获得该数据；没有购买权限或使用其他接口规避限制。已向用户询问已有本地分钟数据或具备权限的配置路径。见 [index_intraday_probe_20260919](../artifacts/evaluation/index_intraday_probe_20260919/README.md)。目前没有分钟模型的性能结论，原目标仍未完成。

后续在项目中检查了 5,487 个不同 CSV/CSV.GZ 文件的数据头，并检查原始参考 ZIP，未找到可用分钟行情。已准备八项日内特征及本地数据导入工具，严格检查时区、完整交易时段、收盘/高低价与日线对齐、成交量/金额单位、缺失和未来数据隔离；16 项软件测试通过。尚未获得真实分钟数据，未训练或回测分钟模型，不能将构建器测试或合成测试数据当作预测性能证据。正式模型及已发布 CSV 保持不变。

## 2026-09-19 事前记录与当前阻塞

重新校验固定资金流向候选 `1b626719...d2d6f1f` 的源码、依赖及种子文件均与冻结版本一致后，基于恢复后的正式 20260918 快照运行一次已有研究入口，未修改模型或门槛。仅增量补取 20260918 个股日线、20260917 资金流向及用于匹配的当日日线（各 5,553 行），以及 20260917 的 SPX/IXIC 各一行；真实收取时间和原始响应哈希均归档。

新 run 为 `67e8b387-86cf-42d6-bb0f-086f92ee451f`。候选实际生成于上海时间 **2026-09-19 05:15:06**，信号日为 20260918、目标交易日为 **20260921**，因此是一条真实开盘前预测，不能回填为 9 月 18 日 18:15 的发布。候选与原模型该日均预测下跌，收益预测同为约 -0.09504%；这条记录本身没有方向纠偏差异。

只读审计显示候选开盘前记录 1 条、已结算 0 条，与原模型共同已结算事前样本仍为 0。原候选的 25/26 历史门槛仍失败，`promotion_allowed` 仍为 false。新记录是后续验证的起点，不是性能达标证据。正式 publication、release、快照及公开 CSV 哈希全部保持原值。见 [prospective_audit_20260919.json](../artifacts/evaluation/moneyflow_price_shadow_v1/prospective_audit_20260919.json)。

分钟数据缺失的同一条件已跨三个连续目标轮次存在：首次接口拒绝、随后本地检索与构建器准备完成、当前再次实测接口仍无权限且没有新数据路径。现有候选未达到完整验收，新增事前预测尚不能结算；没有可据以发布并宣称目标完成的新证据。待已有合法分钟数据/授权配置路径，或后续真实结算等外部条件变化后继续验证，保留原目标，不将其改写为“完成工具”或“获得一条预测”。当前证据汇总在 [direction_bias_status_20260919.json](../artifacts/evaluation/direction_bias_status_20260919.json)。
