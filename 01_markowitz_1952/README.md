# 01｜Markowitz（1952）：均值—方差组合选择

## 一句话理解

不要逐只股票判断“好不好”，而要判断它加入整个组合后，怎样改变组合的预期收益与风险。

## 论文解决了什么问题

设 $w_i$ 是第 $i$ 个资产的权重，$\mu$ 是预期收益向量，$\Sigma$ 是收益协方差矩阵：

$$
\mathbb E[R_p]=w^\top\mu,\qquad
\operatorname{Var}(R_p)=w^\top\Sigma w.
$$

如果只最大化 $w^\top\mu$，资金通常会全部压在预期收益最高的资产上，无法解释分散投资。Markowitz 提出：在给定收益下最小化风险，或在给定风险下最大化收益。

本案例采用长仓、满仓版本：

$$
\begin{aligned}
\min_w\quad &w^\top\Sigma w\\
\text{s.t.}\quad &w^\top\mu\ge r_\text{target},\\
&\mathbf 1^\top w=1,\\
&w_i\ge 0.
\end{aligned}
$$

改变目标收益 $r_\text{target}$，得到一串最优组合；其上边界就是有效前沿。

## 最重要的直觉

- 风险不是各资产波动率的简单加权平均，相关性同样重要。
- 一个单独看起来波动较大的资产，如果和现有组合相关性较低，反而可能降低组合风险。
- 最优权重对预期收益 $\mu$ 很敏感；这正是第二篇 Black–Litterman 要处理的问题。

## 代码做了什么

`markowitz_demo.py`：

1. 用固定随机种子模拟五类资产的三年日收益；
2. 用样本估计年化均值和协方差；
3. 计算等权、全局最小方差和最大夏普组合；
4. 在一系列目标收益下求解有效前沿；
5. 输出权重、指标和有效前沿图。

## 运行

```bash
python markowitz_demo.py
```

输出：

- `outputs/portfolio_weights.csv`
- `outputs/portfolio_metrics.csv`
- `outputs/efficient_frontier.csv`
- `outputs/efficient_frontier.png`

## 建议你亲手改的三个参数

1. 把 `N_OBSERVATIONS` 从 756 改为 126，观察估计误差如何放大。
2. 修改相关系数矩阵中的数值，观察分散化效果。
3. 把优化约束中的 `bounds` 改为允许小幅做空，观察权重是否更极端。

## 论文的局限

- $\mu$ 很难准确估计，而且优化器会放大微小估计误差。
- 方差把上涨与下跌同等视为风险。
- 原始模型忽略交易成本、冲击成本、税、换手和流动性。
- 静态单期模型没有描述今天的交易怎样影响未来调仓。

因此，Markowitz 是“语法”，不是可以原样投入实盘的完整系统。

