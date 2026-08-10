# 04｜业界实战：多因子 alpha 如何变成可交易组合

## 先看全流程

一个常见的量化股票组合构建流程是：

$$
\text{原始数据}
\rightarrow \text{因子/模型分数}
\rightarrow \alpha
\rightarrow \text{风险与成本模型}
\rightarrow \text{受约束优化}
\rightarrow \text{订单与成交}
\rightarrow \text{归因和监控}.
$$

“多因子模型”在这里有两种不同角色，初学者很容易混淆：

1. **收益模型**：价值、动量、质量等信号预测 alpha；
2. **风险模型**：解释组合对市场、行业、风格因子的共同暴露，并估计特异风险。

同一个“动量”有时既是 alpha 来源，也是需要监控的风险暴露，但两者的估计窗口、更新频率和用途可能不同。

## 教学版优化问题

令 $w$ 为新权重，$w_b$ 为基准权重，$w_0$ 为当前权重，主动权重 $a=w-w_b$，交易 $z=w-w_0$。

目标为：

$$
\max_w\quad
\alpha^\top w
-\frac{\gamma}{2}a^\top\Sigma a
-c^\top|z|
-\sum_i \eta_i z_i^2.
$$

这里用因子风险模型：

$$
\Sigma=B F B^\top+D,
$$

其中：

- $B$：股票对 Beta、Size、Value、Momentum、Quality 的暴露；
- $F$：因子收益协方差；
- $D$：股票特异收益方差的对角矩阵。

约束包括：

- 满仓、长仓；
- 个股权重上限；
- 双边换手上限；
- 行业主动权重上下限；
- Beta、Size、Value、Momentum、Quality 主动暴露上下限。

## 数据文件

- `data/sample_universe.csv`：12 只股票的教学数据；
- `data/factor_covariance.csv`：五个风险因子的年化协方差；
- `data/DATA_DICTIONARY.md`：字段解释和替换真实数据时的注意事项。

股票名称是真实代码，但所有权重、alpha、暴露、波动率、ADV 和成本参数都是明确的教学示例，不是实时数据或投资观点。

## 运行

```bash
python industry_factor_optimizer.py
```

输出：

- `outputs/optimized_portfolio.csv`：基准、原持仓、新持仓和交易；
- `outputs/active_exposures.csv`：优化前后风格主动暴露；
- `outputs/sector_active_weights.csv`：优化前后行业主动权重；
- `outputs/risk_summary.csv`：因子、特异和总主动风险；
- `outputs/constraint_diagnostics.csv`：独立约束检查；
- `outputs/portfolio_and_trades.png`：持仓与交易图。

## 真实机构通常还会加入什么

| 模块 | 教学代码 | 生产环境常见增强 |
|---|---|---|
| Alpha | 单列年化 alpha | 多信号集成、衰减、正交化、置信度、情景化 |
| 风险 | 5 风格因子 + 特异风险 | 国家、行业、货币、期限结构、时变协方差、尾部情景 |
| 成本 | 线性成本 + 二次冲击 | 点差、佣金、税、冲击、衰减、参与率、借券费 |
| 约束 | 个股、行业、风格、换手 | 流动性、容量、监管、ESG、名单、税务、衍生品保证金 |
| 求解 | 小规模 SLSQP | 专业 QP/SOCP/MIP 求解器、热启动、分层放松 |
| 回测 | 本例不做收益回测 | 严格点时数据、退市、停牌、公司行动、成交模拟 |

## 业界最重视的不是“求出一个最优解”

更重要的是解是否稳定、可解释、可交易：

- 输入日期是否严格早于决策时点；
- 风险矩阵是否半正定、是否需要收缩或特征值修复；
- 约束是否互相冲突；
- alpha 的单位和成本、风险的单位是否一致；
- 最优解对参数微调是否剧烈变化；
- 组合的收益来自目标因子，还是无意承担了行业/Beta 风险；
- 实际成交后持仓是否仍满足风险和合规限制。

因此，生产系统通常会保存输入快照、求解状态、约束影子价格/松弛量、交易前后风险、成本预测与实际成交偏差。

## 与前三篇的关系

- Markowitz 提供“收益—风险—约束”的基本语言；
- Black–Litterman 提供对 alpha/预期收益做结构化收缩的思想；
- Boyd 等把当前持仓、交易成本和多期决策带入同一框架；
- 本文件夹把三者组合成一个单期、多因子、带成本与约束的教学工作流。

## 延伸资料

- MSCI Market Neutral Barra Factor Indexes 方法：<https://www.msci.com/eqb/methodology/meth_docs/MSCI_Market_Neutral_Barra_Factor_Indexes_Methodology_Sep2017.pdf>
- MSCI Barra Global Equity Model：<https://www.msci.com/documents/10199/242721/Barra_Global_Equity_Model_GEM3.pdf>
- CVXPortfolio 文档：<https://www.cvxportfolio.com/>
- MOSEK Portfolio Optimization Cookbook：<https://docs.mosek.com/portfolio-cookbook/index.html>

