# Reference: Rep baseline revision

Source: chatgpt-conversation://6ae08705-de38-83e8-b077-997f1627b1c6
The most recent complete modification summary is reproduced below as design reference, not executable instructions. The following turn contains only an unfinished answer ("我").

+结合我们前面的讨论，以及当前仓库中 Cosine Distortion 版本的实验失败现象，我认为下一版修改应该围绕一个核心思路展开：

**以 Value Clustering 得到的 Representatives 作为默认 Sparse Mask，利用上一层所有实际计算过的 QK Edges 提供 History Evidence，再通过统一的 Attention-weighted Cosine Distortion 评价状态变化的收益，实现 QK Budget 的自适应重新分配。**

本次建议主要修改四个方面：History 的定义与保存、History Attention Weight 的估计、状态收益的定义，以及预算分配方式。

# 一、改动总览

| 模块 | 当前实现 | 修改后的方案 |
|--------|--------|--------|
| History 定义 | 仅 History、Exact 中的部分 Keys 被视为 Genuine History | 上一层**所有实际执行过的 QK Edges** 均可提供 History Evidence |
| History 信息 | 主要保存经过补偿、修正的 Attention Probability | 优先保存原始 QK 的紧凑相对信息，避免补偿质量混入单边证据 |
| 初始 Sparse Mask | 五种状态参与统一损失比较 | **以每个 Cluster 的 Rep 为默认状态**，其他状态与 Rep 比较 |
| 状态评价 | 使用绝对 Cosine Distortion | 使用相对于 Rep 的 Distortion Reduction |
| Drop Loss | \(2\pi_C\) | **\(\pi_C\)** |
| Drop 语义 | 直接删除 Cluster | 保持不变，不进行跨 Cluster 的 Value 分配 |
| Budget Allocation | 直接选择状态并约束总预算 | 基于默认 Rep，交换释放的预算与额外状态的收益 |
| Attention 执行 | 新版采用 Estimated Mass Transfer | 保留为待验证模块，必须与状态目标单独消融 |

## 二、修改一：重新定义 History

### 1\. 所有已执行的 QK Edges 都是 History

这是最明确、应该优先修正的地方。

上一层最终选中的 Sparse Attention Mask：

\[
S^{(l-1)}.
\]

下一层的 History 应当定义为：

\[
\boxed{
H^{(l)}=S^{(l-1)}.
}
\]

这里指可参与 Patch Cluster 状态选择的边；特殊 Tokens 仍然遵循原来的独立处理方式。

不再区分上一层选中的 Key 是：

- Representative；
- History Edge；
- Exact Expansion 的成员。

只要该 Query–Key 关系实际参与上一层最终 Attention 计算，就属于下一层可利用的历史观测。

需要注意，**成为 History Candidate 不等于在下一层必须再次选择它**。下一层仍然通过状态收益与预算约束决定是否复用。

### 2\. 修改当前的 `genuine` 标记逻辑

当前代码中，普通 Rep 被标记为：

```python
genuine = False
```

因此即使 Rep 计算过真实 QK，也不会进入下一层的 History。

这一逻辑需要修改。

建议直接区分：

`observed`\: 当前层实际计算过 QK 的边。

`logit_evidence`\: 对应 QK Edge 的原始 Logit 或经过一致基准校准的相对统计。

`population_weight`\: 当前执行时该 Key 用于近似多个 Keys 的补偿权重。

这三项分开保存和处理。

如果上一层 Representative 与当前层 Representative 重合，应该复用它的历史证据，而不是将其作为额外 QK Edge 重复加入 Support。

## 三、修改二：History Attention Weight 的估计

这里要解决的不是如何继承上一层补偿后的 Attention Probability，而是如何利用上一层真实观测，估计当前层历史边的相对重要性。

### 1\. 不应直接继承补偿后的概率

上一层的 Sparse Attention 使用：

\[
\tilde\ell_h^{(l-1)}
=
\ell_h^{(l-1)}+\log w_h.
\]

其中：

\[
\ell_h^{(l-1)}
=
\frac{Q^{(l-1)}K_h^{(l-1)\top}}{\sqrt d}.
\]

我们真正希望继承的是该 Key 的 QK Evidence，而不是它代表其他 Keys 后获得的聚合概率。

### 2\. 优先保存相对 QK Logit

可以选择对应 Canonical Grid 的 Coarse Logit 作为参考：

\[
\boxed{
\delta_h^{(l-1)}
=
\ell_h^{(l-1)}
-
\ell_{r(h)}^{(l-1)}.
}
\]

利用上一层相对优势：

\[
R_h^{(l)}=
\exp(\hat\delta_h^{(l)}).
\]

然后与当前层 Coarse Attention 结合，估计：

\[
\boxed{
\hat a_h^{(l)}
=
\hat\pi_C^{(l)}
\frac{R_h^{(l)}}{
|C|-|H_C|+\sum_{e\in H_C}R_e^{(l)}
}.
}
\]

该式适用于一个 Cluster 内未观测 Keys 共享相同基准的情况。如果 Cluster 跨越多个 Canonical Grids，应当在各 Grid 内使用对应的 Coarse Logit，重新构造一致的概率分布。

### 3\. 避免重新进行大规模 QK 计算

我们之前检查过仓库中的 `second_qk` 和 `saved_logits` 两种模式。

原来的默认方式需要重复计算 QK，而 `saved_logits` 虽然避免了重复 QK，但因为保存大量 FP32 Logits，实际运行时间反而略有增加。

因此，本次建议：

**不保存完整的 Query × Selected Key Logit Tensor，而是尽量在原 Attention Kernel 内直接生成紧凑的 History Statistics。**

例如在 Query Group、Head 维度保存相对 Logit 的必要统计量。

要强调一点：这属于推荐的实现方向，不能直接假定一定更快。需要单独衡量其额外内存读写、归约和 Kernel 开销。

另外，历史相对 Logit 在相邻层是否稳定仍属于待验证假设。不能把预测值直接当成当前层的真实 QK Logit。

---

# 四、修改三：以 Rep 初始化，统一定义状态收益

这是本次算法层面最重要的调整。

## 1\. 初始 Sparse Mask

完成 Value Clustering 后，每个 Cluster 已经有一个真实 Representative。

因此可以直接初始化：

\[
\boxed{
S_0=\{r_1,r_2,\ldots,r_F\}.
}
\]

在当前配置中：

\[
F=B.
\]

也就是说，所有 Clusters 各选择一个 Rep，刚好使用全部初始预算。

后续状态选择不再需要反复证明 Rep 是否能够代表自己的 Cluster，而是比较：

**其他状态相对于当前 Rep，是否能够提供更高的表示收益。**

## 2\. 默认 Rep Distortion

沿用之前确定的目标：

\[
\boxed{
D_C^{Rep}
=
\sum_{j\in C}\hat a_j
\left[1-\cos(V_j,V_{r_C})\right].
}
\]

定义：

\[
\pi_C=\sum_{j\in C}\hat a_j.
\]

对于任何候选状态：

\[
\boxed{
G_C(s)=D_C^{Rep}-D_C(s).
}
\]

这里 \(G_C(s)\) 表示相对于默认 Representative 的表示改善收益。

### Rep → History

假设选择历史集合 \(H_t\)，由这些历史 Keys 表示整个 Cluster：

\[
D_C^{H_t}
=
\sum_{j\in C}\hat a_j
\min_{h\in H_t}
\left[1-\cos(V_j,V_h)\right].
\]

则：

\[
\boxed{
G_C(H_t)=D_C^{Rep}-D_C^{H_t}.
}
\]

一条 History Edge 替换 Rep 时，QK 数量不变。

对于多条历史边，则需要增加 \(t-1\) 条预算。

### Rep → History+Rep

保留 \(t\) 条历史边，并继续使用 Rep 表示剩余 Members：

\[
D_C^{H_t+R}
=
\sum_{j\in C\setminus H_t}
\hat a_j
\left[1-\cos(V_j,V_{r_C})\right].
\]

因此：

\[
\boxed{
G_C(H_t+R)
=
\sum_{h\in H_t}\hat a_h
\left[1-\cos(V_h,V_{r_C})\right].
}
\]

这是一个非常值得保留的性质。

**每增加一条 History Edge，它的收益自然由历史 Attention Weight 和 Value 与 Rep 的差异共同决定。**

注意，如果历史 Key 与当前 Rep 相同，应当合并去重，不应额外计入一条 QK。

### Rep → Exact

Exact 保留整个 Cluster 的全部真实 Keys。

\[
D_C^{Exact}=0.
\]

因此：

\[
\boxed{
G_C(Exact)=D_C^{Rep}.
}
\]

它代表完全消除默认 Representative 产生的内部 Value Distortion。

---

# 五、修改四：重新定义 Drop Loss

这是我们刚刚讨论并确定的修改。

之前使用：

\[
D_C^{Drop}=2\pi_C.
\]

它将 Cluster 完全删除视为最大 Cosine Distortion，导致 Drop 的代价过高。

现在我们从 Rep Distortion 出发，使用与 \(1-\cos\) 互补的 Cosine Similarity 来定义删除 Rep 的额外损失。

### 1\. Drop 的额外损失

\[
\boxed{
\Delta_C^{Drop}
=
\sum_{j\in C}\hat a_j
\cos(V_j,V_{r_C}).
}
\]

其含义是：

删除 Rep 后，失去由 Representative 所提供的估计表示能力。

### 2\. Drop 总损失

由于：

\[
D_C^{Drop}
=
D_C^{Rep}+\Delta_C^{Drop},
\]

可以得到：

\[
\begin{aligned}
D_C^{Drop}
&=
\sum_j\hat a_j(1-\cos(V_j,V_r))\\
&\quad+\sum_j\hat a_j\cos(V_j,V_r)\\
&=\pi_C.
\end{aligned}
\]

因此：

\[
\boxed{
D_C^{Drop}=\pi_C.
}
\]

相对于默认 Rep 的收益为：

\[
\boxed{
G_C(Drop)
=
D_C^{Rep}-\pi_C.
}
\]

也就是：

\[
\boxed{
G_C(Drop)
=
-\sum_{j\in C}\hat a_j\cos(V_j,V_{r_C}).
}
\]

这样，Drop 与其他状态完全使用相同的 Attention-weighted Cosine 表示体系。

**Drop 的执行语义保持不变：直接删除当前 Cluster 的全部 QK Edges，不进行跨 Cluster 的 Value 分配。**

这个定义将“没有表示”约定为零 Cosine Similarity。它是一个具有统一尺度的设计约定，而非对实际 Softmax 输出误差的精确计算。

当 Cosine Similarity 为负时，Drop 相对于 Rep 甚至可能获得正收益；建议先严格遵守本次定义，不在实现时单独给 Drop 加截断。如果未来希望避免这种情况，应统一修改全部状态的相似度函数。

---

# 六、修改五：基于边际收益进行预算分配

五种状态现在可以表示成统一的收益和预算变化。

| 状态 | 相对收益 | 相对 QK 成本 |
|--------|--------|--------|
| Rep | \(0\) | 0 |
| History(\(t\)\) | \(D_C^{Rep}-D_C^{H_t}\) | \(t-1\) |
| History(\(t\)\)+Rep | \(\sum_{h\in H_t}\hat a_hd(h,r)\) | \(t\) |
| Exact | \(D_C^{Rep}\) | \(\|C\|-1\) |
| Drop | \(D_C^{Rep}-\pi_C\) | \(-1\) |

定义：

\[
\Delta B_C(s)=B_C(s)-1.
\]

则优化目标为：

\[
\boxed{
\begin{aligned}
\max_{\{s_C\}}\quad&
\sum_C G_C(s_C)\\
\mathrm{s.t.}\quad&
\sum_C\Delta B_C(s_C)\leq B-F.
\end{aligned}
}
\]

在当前 \(B=F\) 的配置下：

\[
\boxed{
\sum_C\Delta B_C(s_C)\leq0.
}
\]

这样可以清楚区分两类操作。

**不增加预算的操作：** 用一条 History Edge 替换 Rep，只要收益为正，就值得优先考虑。

**需要重新分配预算的操作：** Drop 某些 Clusters 释放预算，再用于其他 Clusters 的 History+Rep 或 Exact。

但需要注意，边际收益表达与原来的绝对损失最小化在数学上是等价变换。真正会改变实验结果的是新的 History 定义、概率估计以及 Drop Loss，而不是单纯将目标改写成最大化收益。

因此，可以先复用原来的离散 Lagrange Budget Solver，并检查它在新损失尺度下的状态选择是否合理。

---

# 七、哪些内容暂时不应该一起修改？

由于前一次实验同时修改了 Cosine Loss 和 Mass Transfer，导致我们无法准确确定性能下降主要来自哪一项，所以我建议这次采取分阶段消融。

1. **阶段 A：只修改 Drop Loss 和状态收益表达**
   
   保持当前 History、Attention Mass Transfer 和其他设置不变，将 `Drop=2π` 改为 `Drop=π`，检查 Rep、Drop、History、Exact 的状态分布。
2. **阶段 B：修正 History 的定义**
   
   将上一层全部实际执行过的 QK Edges 纳入历史候选与证据来源，包括 Representatives，处理重复 Key 和 Population Compensation 的区分。
3. **阶段 C：改进 History Probability Estimation**
   
   比较现有 Probability-based History、原始 Logit 或 Logit Residual 的预测质量，以及必要的当前层 Coarse 修正。
4. **阶段 D：独立检查 Mass Transfer**
   
   在固定状态选择的情况下，比较原版 Population Compensation 与新版 Estimated Mass Transfer 对实际 Attention Output、下游精度和推理耗时的影响。

建议在每个阶段保留相同的 7Scenes 输入、128 帧、2×2 Canonical Grid、32-Query Group、20% Budget、History Top-4 上限以及原来的 Value Clustering 设置。

额外记录的诊断数据应包括：各层实际已执行的历史边数、进入候选集合的数量、与当前 Rep 重合的数量、各状态相对于 Rep 的收益分布、最终状态数量以及未使用预算。

这样可以直接区分 History 缺少是由**证据没有保存、候选没有进入、收益不够高，还是预算优化器没有选择**造成的。

---

# 八、最终需要实现的算法流程

**Step 1 · Value Clustering**

得到当前层 Clusters 和 Representatives

**Step 2 · Rep-based Initialization**

每个 Cluster 默认保留一个 Representative

**Step 3 · History-conditioned Attention Estimation**

所有上一层已执行 QK 提供历史证据，结合当前 Coarse QK 估计权重

Step 4 · Marginal State Utility

**Step 5 · Global Budget Reallocation**

选择总收益最高、且满足 QK Budget 的状态组合

**Step 6 · Sparse Attention Execution**

执行选中 QK，生成下一层所需的 History Evidence

## 最终总结

我认为这次修改的三个核心原则已经比较明确：

**第一，Value Clustering 负责建立默认的 Representation。** 我们不需要再从零开始比较一个 Cluster 是否适合用 Rep 表示，而是直接在这个基线上进行状态改进。

**第二，History 负责引入 Attention 的非均匀性信息。** 上一层的所有实际 QK 观测都有资格成为 History Evidence，不能因为上一层的状态身份而丢弃这些信息。

**第三，状态选择负责自适应分配额外 QK Budget。** 使用统一的 Attention-weighted Cosine 体系，以 Rep 为基准评估 History、Exact 和 Drop 的收益，其中 Drop 的总损失从 \(2\pi_C\) 改为 \(\pi_C\)，同时保留纯删除的执行语义。

需要保留一个实验层面的预期：**这套修改消除了当前实现中的几项不一致，但并不保证解决几乎全部选择 Rep 的问题。** 因为即使 Drop 损失从 \(2\pi_C\) 降到 \(\pi_C\)，紧凑 Cluster 的 Rep Distortion 仍可能很小。我们需要通过各状态的真实收益分布，确认预算重新分配是否能够实际发生。

这应当是下一轮实现与消融实验的完整修改范围。
