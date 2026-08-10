# 组合优化经典论文学习包

这个项目面向第一次系统学习量化组合优化的读者。三篇论文按“静态基础 → 稳健预期收益 → 动态可交易组合”递进，最后用一个接近业界研究流程的多因子案例把它们串起来。

## 你会学到什么

| 顺序 | 论文 | 核心问题 | 代码重点 |
|---|---|---|---|
| 1 | Markowitz (1952), *Portfolio Selection* | 怎样同时考虑收益、风险和相关性？ | 有效前沿、最小方差、最大夏普 |
| 2 | Black & Litterman (1992), *Global Portfolio Optimization* | 样本均值太吵、权重太极端怎么办？ | 反向优化、贝叶斯融合、观点置信度 |
| 3 | Boyd et al. (2017), *Multi-Period Trading via Convex Optimization* | 真正调仓时怎样加入成本、约束和未来计划？ | 单周期与多周期优化、交易成本、滚动决策 |
| 4 | 业界工作流 | 多因子策略如何从 alpha 变成可成交持仓？ | 因子风险模型、行业/风格约束、换手和成本、诊断 |

## 最推荐的学习顺序

1. 先读每个文件夹的 `README.md`，暂时不要追求看懂论文的每个公式。
2. 运行代码，先观察输入、最优权重和输出图。
3. 修改一个参数，例如风险厌恶系数、观点置信度或换手上限，再运行并比较。
4. 回到论文，重点读“问题设定、目标函数、约束、实验结果”。
5. 最后阅读 `04_industry_workflow`，理解论文模型怎样进入真实策略管线。

## 环境与运行

建议使用 Python 3.10 或以上版本。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python run_all.py
```

运行后，每个子文件夹的 `outputs/` 会生成 CSV 和 PNG。压缩包中已经附带一次固定随机种子的示例运行结果。

## 先记住这一个统一公式

多数实务组合优化都可以写成：

$$
\max_w\quad
\underbrace{\alpha^\top w}_{\text{预期收益}}
-\underbrace{\lambda\,w^\top\Sigma w}_{\text{风险惩罚}}
-\underbrace{C(w-w_{\mathrm{old}})}_{\text{交易成本}}
$$

并加入预算、个股、行业、风格暴露、换手、流动性等约束。三篇论文的差异，主要在于如何得到 $\alpha$、如何描述 $\Sigma$、是否显式处理交易和未来多个时期。

## 文件导航

- `01_markowitz_1952/`：所有现代组合优化的起点。
- `02_black_litterman_1992/`：解决“预期收益输入不可靠”的经典办法。
- `03_boyd_multiperiod_2017/`：把组合优化升级为动态交易决策。
- `04_industry_workflow/`：教学版业界多因子组合构建器。

## 重要说明

- 示例数据为确定性模拟数据或明确标注的教学数据，不代表真实证券观点。
- 代码重点是透明和可读，不追求低延迟或大规模生产性能。
- 这不是投资建议。真实资金使用前，需要做点时数据处理、幸存者偏差控制、容量评估、交易模拟、合规审查和独立风险验证。
