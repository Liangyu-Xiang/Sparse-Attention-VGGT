# Shared Value Query-256：状态损失与两种 History 语义

当前用户选定的History定义为 **H ∪ Rep**，具体定义、成本与权重见 [History支持定义](shared_value_history_union_rep.md)。对应现有A实现。

本分支是实验实现，不替换原默认配置。保持 `history_query_interval=256`、原来每32个 Query 共享支持集、原聚类和16轮预算求解。总 Key 上限为20%，包含每帧始终保留的5个 Special Key；patch 预算为 `floor(0.2*N)-N_special`。不通过提高保留率或强制 History 配额改善结果。

## 概率与输出特征

对每个 Query 组和当前簇 C，已有粗先验给出簇质量 π_C。有真实历史采样信息的旧支持成员保留其已有概率 p_j；其余成员均分剩余质量：

\[
\hat p_j = \frac{\max(0,\pi_C-\sum_{i\in K_C}p_i)}{\max(1,|C|-|K_C|)},\quad j\notin K_C.
\]

未采样 Query 组全部按簇内均匀概率处理。这里的已知 p 来自上一层既有历史证据，不计算当前簇所有成员的 QK。离线 Dense QK 和独立 Dense 前向只用于 mask 确定之后的诊断，未放入生产分配器。

损失特征为 `y_j = concat_heads(V_j) W_o^T R`（以列向量记号需相应转置），R 是固定种子的8维正交随机投影；然后用当前层 patch 特征的 RMS 范数统一归一化。这避免先对 head 求均值时正负抵消，但仍是输出空间的近似，不是完整1024维输出或每个 head 的精确损失。

令 μ 为包含 Special 的估计全局注意力输出，m_C=Σ_C p̂_j y_j。中心化 Drop 损失为：

\[
L_D(C)=\sum_{j\in C}\hat p_j\|y_j-\mu\|^2.
\]

采用 μ 的原因是：softmax 重新归一化后，移除成员的影响取决于它相对输出的偏移，而不只取决于它的 Value 模长。该式仍是可分配的二阶代理，不能保证消除跨簇误差的交叉项。

## A：History 只表示自身，剩余成员由 Rep 表示

H 为可用的旧支持成员；若包含当前 Rep r，将 r 从普通 History 集合排除。普通历史成员的执行权重严格为1；Rep 的执行权重为：

\[
w_r=\frac{\pi_C-\sum_{j\in H\setminus\{r\}}\hat p_j}{\hat p_r}.
\]

Rep 状态只保留 r，权重 π_C/p̂_r。History 状态保留 `(H\{r}) ∪ {r}`，其成本为普通 History 数+1。

\[
L_R(C)=\sum_{j\in C}\hat p_j\|y_j-y_r\|^2,
\quad L_H(C)=\sum_{j\in C\setminus(H\setminus\{r\})}\hat p_j\|y_j-y_r\|^2,
\quad L_E(C)=0.
\]

配置：`state_loss_centered_residual_energy.json`，operator=`residual_rep`。History 只有当前 Rep 时，H 和 R 支持、权重、损失、成本都一致。注意：这仍然允许显式 Rep 承担剩余簇质量，不是完全不做质量补偿。

## B：History 集合表示整个簇

保留的 History 集合 H 共同承担簇质量，所有历史成员使用相同权重 `π_C / Σ_H p̂_j`。其估计条件均值为：

\[
\bar y_H=\frac{\sum_{j\in H}\hat p_j y_j}{\sum_{j\in H}\hat p_j}.
\]

History 损失只评价未保留成员由该均值表示的误差：

\[
L_H(C)=\sum_{j\in C\setminus H}\hat p_j\|y_j-\bar y_H\|^2.
\]

Rep 损失同 A；Drop 使用中心化式；Exact 为零。配置：`state_loss_group_centered.json`，operator=`group_history`。

两个边界都必须成立：H={r} 与 Rep 完全等价；H=C 与 Exact 完全等价。若对 H 内的成员也重复计其到均值的方差，会破坏第二个边界，本实现不这样做。空集或估计质量为零的 History 不可用。

簇内完全均匀时，Rep 权重退化为 n，B 的 History 权重退化为 n/|H|；A 的 Rep 权重退化为 n-|H\{r}|，其余 History 权重为1。未采样组使用整数比值查表，有历史采样的组才使用概率比值除法。

## 已比较的损失

共34种组合，完整结果见 `outputs/state_loss_search_20261105/loss_report/report.md`：

- 单位/Rep人口权重两套执行方式，各7种损失：原点能量、中心化能量、概率平方能量、中心化矩误差、两种质量惩罚、条件均值+质量误差。
- A 两种历史门控，各5种损失：原点/中心化残差能量、矩误差、矩+Drop质量项、矩+方差项。
- B 两种历史门控，各5种损失：未保留成员到History均值的失真、中心化Drop版本、History混合方差版本、矩误差、矩+方差版本。

门控分别是旧规则（历史概率高于簇均值或为当前Rep）和所有已观察旧支持；当前生产对照用后者。它只扩大旧支持内的候选集，不增加实际预算。跨系列比较同时改变了概率假设、特征、执行语义、门控，不能把所有改善归因于单独一条损失公式。

直接的簇输出矩损失也测试过：`||m_C-π_C y_r||²` / `||m_C-π_C ȳ_H||²`，Drop 为 `||m_C-π_C μ||²`。在两个固定特征场景上，其中间层输出误差高于所选能量形式，并存在预算略未用满的问题，因此没有凭“公式更接近输出”就直接采用。

## 边界与验收

未知 QK 差异无法从 Value 完全恢复；簇内均匀是明确假设。权重由估计概率得到，而执行 attention 使用实际当前 logits，因此估计簇质量一致也不等于实际 Dense 概率严格一致。8维投影、簇间误差相加、层间误差累积都可能造成残差，必须分别测量。

CPU 代数测试和 CUDA 测试检查：同支持同权重同损失、History全簇与Exact等价、均匀余量守恒、GPU统计/全部已实现损失与参考式一致、执行权重、支持唯一、图缓存当前输入和常量更新。两种最终候选在读取新 pose/reconstruction 质量结果前冻结，见 `quality/frozen_manifest.json`。

生产配置没有 Dense attention 依赖。质量实验使用 pose+recon_unposed 六项指标；所有诊断六项标 N/A。局部逐层对比与独立 Dense 轨迹的累计对比分开报告。History 状态比例与真正复用旧 Key 的比例分开报告。

## 新确认的结构问题：簇损失相加不等于输出误差

即便完全接受未知成员均匀，而且假设该概率恰好等于真实概率，单独计算每簇损失后相加仍可能选错 mask。这不是依赖“再去估计未知QK”的论点。

设 Dense 的输出为 μ，全部概率归一化为1。某簇状态 s 执行后的未归一化质量/Value矩分别为 b_C(s)、a_C(s)，原簇矩为 m_C，则定义：

\[
\delta_C(s)=m_C-a_C(s)-(\pi_C-b_C(s))\mu.
\]

所有 Special 始终保持时，真正的全局输出误差是：

\[
\|\mu-\tilde\mu\|^2
=\frac{\|\sum_C\delta_C(s_C)\|^2}{(1-\sum_C\pi_C+\sum_C b_C(s_C))^2}.
\]

现有逐簇矩损失相加只保留 `Σ||δ_C||²`。它遗漏了不同簇误差向量的交叉项，也没有按整张 mask 的最终 softmax 分母统一评价；能量代理还多一层二阶近似。这个问题在 A 和 B 中都未解决。

可复现反例：两个簇的 Value 分别为 `[0,2]`、`[0,-2]`，各以首成员0为Rep；另有一个 Value=1 的 Special。5个 token 概率都严格为0.2，因此 Dense 输出为0.2。比较同为2个patch Key的两张mask：

| mask | 两簇的逐簇中心化能量之和 | 两簇的逐簇矩损失之和 | 实际输出 | 与Dense绝对误差 |
|---|---:|---:|---:|---:|
| Rep + Rep，权重均为2 | 1.6000 | 0.3200 | 0.2000 | 0 |
| Exact第一簇 + Drop第二簇 | 0.9760 | 0.2304 | 1.0000 | 0.8000 |

两种可分损失都会更喜欢后者，然而前者精确恢复Dense。该反例保持统一概率和相同预算，不计算任何额外QK，说明“单个状态损失语义正确”仍不足以保证最终mask的输出正确。测试 `test_separable_loss_can_prefer_wrong_mask_even_with_uniform_truth` 已覆盖。

它证明了结构缺陷的存在，没有测出该缺陷对真实数据精度损失的贡献比例。不能把全部剩余退化直接归因于它；跨层传播、先验误差、输出投影近似和共享mask仍可能共同作用。

## 联合目标的实际尝试

`joint_state_loss_reference.refine` 以 B 的可分解为起点，使用相同概率/8维输出特征，直接评价整张mask的 `||Σδ||²/Z²`，每次检查所有簇的4种状态，最多作32次预算可行的单簇更新。还测试了 γ=0.01/0.1 的可分能量正则。选择过程仍不接收 Q/K。

在同一冻结输入特征中，Kicker中间层（10–16）的纯联合目标降到初值的18.25%，但完整Wo空间实际Dense RelL2从13.020%升到13.634%；Playground中降到13.80%，实际RelL2从12.170%升到12.574%。γ=0.1基本与可分控制持平，没有在两个场景同时得到明确收益。详见 `joint_report/report.md`。

所以已经尝试修正跨簇目标，而不是停留在反例分析；结果同时说明当前代理概率/输出投影与真实输出仍存在错位，不能只按代理损失下降判断修复成功。这组实验没有区分概率近似与8维投影各自造成多少偏差。该原型只有冻结特征诊断，不作为完整质量结果，也不接入默认生产路径。

同一缺陷也已在严格20%预算下复现：35个token中，5个Special始终保留，30个patch分为大小2与28的两个簇，C=B=2；5个Special的Value为`[1,0,0,0,0]`，第一簇`[0,2]`，第二簇`[0,-2,0,…,0]`。所有35个概率完全相等。两张mask都只保留2个patch+5个Special，恰为20%。Rep+Rep输出为1/35，精确匹配Dense；Exact+Drop输出为3/7，绝对误差0.4，但中心化能量和可分矩损失仍都偏好后者。测试 `test_separable_misranking_also_holds_at_twenty_percent_with_five_specials` 覆盖这一版本，原始数值见counterexample JSON的`twenty_percent_case`。
