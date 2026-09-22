# 测试与冻结研究证据

在匹配模型清单的 Python 环境中安装测试依赖，然后运行软件回归：

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

默认运行不依赖本机研究归档的软件测试，包括真实模型的因果性测试、服务发布与权限测试、冻结模型包逐文件哈希测试，以及候选证据加载器的成功、缺失文件、合同不符和篡改拒绝测试。

两项历史研究验收单独标记为 `research_artifacts`。它们需要从原研究机器或可信备份恢复原始证据；目录被 Git 忽略，不能用合成测试数据代替，也不能把未运行的验收记录为通过：

- `artifacts/evaluation/bilstm_causal_v1/`：`contract.json`、`summary.json`、`bilstm_causal_predictions.csv`、`bilstm_live_replay_parity.json`。
- `artifacts/evaluation/ensemble_moneyflow_price_downside_v1/`：原候选冻结工具要求的完整研究输入和验收记录。

恢复文件后显式执行：

```sh
python -m pytest -q -m research_artifacts
```

指定研究验收时，缺失文件会使检查失败；不会自动抓取或重新生成历史成绩。加载器继续核对固定输入哈希、模型合同、逐日预测哈希与历史实时/回放一致性证据。软件回归通过与历史研究验收通过分别记录。

冻结运行包及参与指纹的源码保留原字节；`.gitattributes` 禁止对这些文件自动转换换行。`tests/test_frozen_release_files.py` 在检出后核对清单。不得为使测试通过而修改清单哈希。

服务发布演练使用 SQLite backup 产生的隔离数据库；当前四模型的实际重算验收独立于以上两个研究 fixture。
