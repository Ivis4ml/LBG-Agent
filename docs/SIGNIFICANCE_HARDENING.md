# Significance Hardening · Stage 1.5 plan

> Status: planning · drafted 2026-05-27 · driver branch `codexreview`

## Why this exists

Discovery accepts factors via the validation gate at α=0.20 LCB; H1 验收以 sealed
95% CI 下界 > 0 为标准。但实际我们跑了几十次 trial、最终库里 ~11 张卡片之间存
在**搜索带来的选择偏倚**："被 gate 接受"并不等于"在搜索结构调整后仍显著"。
此外 PROPOSAL §20 item 32 写了 Benjamini-Hochberg FDR 控制 0.10，但代码里没有
实现（`grep -rn "fdr\|benjamini\|multipletests" lbg/verdict/` 为空）。这是一个
**规范空洞**，必须先填上才能在此基础上叠加更精细的家族级检验。

Advisor 还指出一个 LBG 特有的结构性限制：LLM 不是在随机空间搜索，而是在其训练
得到的因子先验上搜索；因此任何依赖"如果它真的在乱猜会怎样"的论证都需要明说这
一前提。同时 SPY 2021-2024 对 LLM 而言不是真正的 OOS（其训练数据包含该时期的
因子文献），真正干净的 OOS 只能由 Stage 2 的 paper trading（>2024-12-31 数据）
提供。这两条限制无法通过统计手段补救，只能在最终报告中明确承认。

## Outline · five steps

### Step 1 · BH FDR per-card 校正（半天）· DONE (commit 4de067a)

- **目的**: 落实 PROPOSAL §20 item 32 —— 在所有被接受的卡片上控制 FDR 0.10。
- **实现**:
  - `lbg/verdict/bootstrap.py` 增加 `p_value_one_sided` 输出字段
    （= bootstrap 差值分布中 ≤ 0 的比例，等价于 percentile-bootstrap 单边 p）
  - 新建 `lbg/verdict/multiple_testing.py`，提供
    `benjamini_hochberg(p_values, q=0.10)` 返回每张卡片的 `reject/keep` 标记
  - `PerCardValidationResult` 新增字段 `p_value_one_sided` 与 `bh_validated`
  - `compute_per_card_sealed_validation` 在算完单卡 CI 之后对家族应用 BH，
    把 `bh_validated` 写入卡片的 `evidence.sealed_summary`
  - `lbg/verdict/h1.py` 的 `validated_factor_count` 改成 BH-validated 的计数
    （CI 下界 > 0 退化为辅助列，保留向后可读性）
- **测试**:
  - 全零假设输入（diffs 平均为 0）应当几乎无拒绝
  - 全强假设输入（diffs 全为正）应当全部拒绝
  - 一个已知的 5-中-2 显著合成案例验证 BH 排序
- **退出条件**: `pytest tests/test_per_card_sealed_validation.py` 与新
  `tests/test_multiple_testing.py` 全绿；H1 报告中 `validated_factor_count`
  以 BH 后的计数为准；当前 11 张卡片重跑后给出 BH 校正后的数字（预期会下降）。

### Step 2 · Hansen 2005 SPA 家族级检验（1-2 天）· DONE (commit e8d4e65)

- **目的**: 直接回答"我手上这 N 张卡片，作为家族，至少一张真显著吗"。
  与 BH 互补：BH 控 per-card FDR，SPA 控 family-wise 一致性 + 候选间相关结构。
- **实现**:
  - 新建 `lbg/verdict/spa.py`，实现 stationary-bootstrap（Politis & Romano 1994，
    geometric block length，期望 ≈ 10）+ Hansen 2005 的 SPA_c / SPA_l / SPA_u
    三种 p 值（保守 / 共识 / 自由），默认输出 SPA_c
  - 输入：N 张卡片相对 buyhold baseline 的日 ΔSharpe 序列矩阵（T × N）
  - 输出：`spa_p_value`、`best_candidate_id`、`recentering_term` 等诊断
  - `lbg/verdict/h1.py` 增加 `spa_verdict` 字段，与 BH 计数并列
- **测试**:
  - 全零假设输入应当 SPA p ≈ U(0,1) 分布
  - 单一强信号 + 多个噪声应当 SPA 拒绝且识别正确的 best candidate
  - 与 BH 在同一组输入上的对比测试，证明两者答的是不同问题
- **退出条件**: 当前 11 张卡片重跑后给出 SPA p 值；如果 SPA 拒绝 → 家族级证据；
  如果不拒绝 → BH 拒绝的子集只能解读为"个别可能显著，整体不显著"。

### Step 3 · Deflated Sharpe Ratio 单卡片调整（半天）· DONE

- **目的**: 在 per_card 表格里加一列"考虑搜索 N 次后这张卡片的 Sharpe 是否真 > 0"。
  作为 BH/SPA 之外的第三种校正视角（Bailey & López de Prado 2014）。
- **实现**:
  - 新建 `lbg/verdict/dsr.py`，实现 DSR(SR̂, T, γ_3, γ_4, N)，
    其中 γ_3 γ_4 用 sealed 窗口实测偏度峰度，N 为该 campaign 接受的卡片总数
    （保守起见也支持用 attempted trial count 作为 N 的上界）
  - `PerCardValidationResult` 新增字段 `deflated_sharpe_ratio`、`dsr_threshold`
  - 报告里和 pathwise CI、BH 标记并列展示
- **测试**:
  - 已知 SR̂, T, N 的 closed-form 验证（对照 Bailey 论文表 1）
  - 收益率正态化下偏度峰度 → 0 时 DSR 退化为标准 Sharpe z-检验
- **退出条件**: 当前 11 张卡片给出 DSR 值；与 BH/SPA 形成三列校正视图。

### Step 4 · 合成 null 标定（1-2 周，含算力）

- **目的**: 经验测量"在真零假设下 gate 接受率 / sealed CI 通过率"，
  作为整套 pipeline 的 operating characteristic。
- **实现**:
  - 新建 `scripts/synthetic_null_calibration.py`，
    使用 stationary-bootstrap 重排 SPY 收益（保留收益率边际分布与短期自相关）
  - 在每个重排数据集上跑 campaign（budget=15、iterations=2），共 30 次
  - 汇总：接受率、单卡 CI 通过率、BH 拒绝率、SPA 拒绝率的经验分布
  - 与真实数据上的对应指标做对比
- **报告中必须明说的 LLM-prior caveat**（advisor 指出）:
  > 合成 null 上的接受率并不等于 pipeline 在随机空间的 false-discovery rate。
  > LLM 仍然基于训练先验提出 ADX、MACD 这类已知因子，因此 effective N 小于名义
  > 试验次数；合成 null 上的高接受率既可能说明 gate 过于宽松，也可能说明 LLM
  > 先验与 bootstrap 伪影部分重合。本节给出的是 pipeline 在 LLM 先验加随机数
  > 据条件下的 operating characteristic，而非纯统计意义上的 FDR。
- **测试**:
  - synthetic-null 脚本可以在 mini 模式（budget=2、iterations=1）3 分钟内跑通
  - 标定输出 JSON schema 稳定，可被报告生成器消费
- **退出条件**: 拿到 30 次重排 campaign 的接受率分布；与真实数据接受率对比的
  z-score 或经验百分位写入最终报告。

### Step 5 · Walk-forward 多窗口 sealed（1-2 周）

- **目的**: 把 2021-2024 切成 4 个不重叠 1 年片段，每张卡片在每个片段上独立打分。
  在 ≥3 个片段上 CI 下界 > 0 的因子是真正强证据。
- **实现**:
  - 修改 `SealedVault` 支持多窗口写入（当前只允许一次写入；这一改动需要扩展
    schema 而非放宽 invariant）
  - 修改 `compute_per_card_sealed_validation` 接受窗口列表，输出每张卡片在每个
    窗口的 CI 与 BH 标记
  - 报告中给出"窗口 × 因子"矩阵：单元格颜色按 CI 下界正负染色
- **测试**:
  - 4 窗口 schema 的 round-trip 测试
  - 已知合成因子在 4 窗口上的稳定性表现验证矩阵展示正确
- **退出条件**: 当前 11 张卡片产出 4×11 矩阵；论文里报"在 ≥3 窗口稳定显著的
  因子数"作为最强陈述。
- **风险**: 这一改动触及 PROPOSAL §5 的"密封测试恰好打开一次"条款，需要在
  PROPOSAL 中先注脚说明"多窗口 = 一次性切分，每个子窗口仍只评一次"，避免被读
  作放宽密封纪律。

## Sequencing summary

| 步骤 | 工作量 | 依赖 | 何时启动 |
|------|--------|------|----------|
| 1 BH FDR | 半天 | — | **立即** |
| 2 SPA | 1-2 天 | 1 完成 | 1 之后 |
| 3 DSR | 半天 | 1 的报告框架 | 与 2 并行可 |
| 4 合成 null | 1-2 周 | 1+2+3 完成 | 1-3 拿到真实数据基线后 |
| 5 Walk-forward | 1-2 周 | PROPOSAL 注脚 | 1+2+3 之后；可与 4 并行 |

## Out of scope（写论文时必须承认的限制）

- **SPY 2021-2024 不是真 OOS**：LLM 训练数据覆盖该时期因子文献。真正干净的 OOS
  只有 Stage 2 paper trading 用 >2024-12-31 数据。这一条不可由统计方法补救。
- **多重资产泛化**：本计划全部限于 SPY 一个标的（PROPOSAL §20 item 19）。
  跨资产泛化是未来里程碑。
- **因果识别**：本计划只验证统计显著性，不主张任何"因子→收益"的因果机制；
  机制讨论留给 dossier 的 `core_analysis` 字段，不进入 H1 裁定。

## References

- Benjamini, Hochberg 1995, "Controlling the False Discovery Rate"
- Hansen 2005, "A Test for Superior Predictive Ability"
- Politis, Romano 1994, "The Stationary Bootstrap"
- Bailey, López de Prado 2014, "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality"
- White 2000, "A Reality Check for Data Snooping"
