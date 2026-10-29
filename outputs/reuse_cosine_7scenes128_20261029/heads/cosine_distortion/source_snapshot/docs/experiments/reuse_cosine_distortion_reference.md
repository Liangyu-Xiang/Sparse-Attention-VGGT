# Source

User-referenced chat: chatgpt-conversation://6ae08705-de38-83e8-b077-997f1627b1c6
Latest objective/state definition, recorded as reference data.

我理解你的意思了。你希望衡量的不是某个状态能否复现一个估计的 Dense Attention Output，而是：

**在给定当前可用 Attention 信息的情况下，该状态选中的 Keys 能够在多大程度上表示整个 Cluster 的 Values。**

因此，状态损失应该直接建立在 **Attention-weighted Cluster Distortion** 上，而不是建立在 Cluster Mean Contribution 的差值上。

我认同这一方向。此前的均值误差可能因为向量抵消而等于零，即使 Cluster 内的 Values 实际上具有很大的离散性。基于逐 Token 的 Cosine Distortion，可以避免这种问题。

## 一、首先定义统一的 Cluster Distortion

设 Cluster \(C\) 包含 \(n\) 个 Tokens，各 Token 的归一化 Value 为：

\[
z_j=\frac{V_j}{\|V_j\|_2}.
\]

定义两个 Values 之间的 Cosine Distortion：

\[
\boxed{
d(i,j)=1-z_i^\top z_j
}
\]

这里需要纠正一个小细节：是 **Cosine Distortion 等于 0**，而不是 Cosine Similarity 等于 0，才表示两个 Values 方向完全一致。

对于一个状态 \(s\)，假设它选择的 Key 集合为 \(S_C(s)\)，并由 \(r_s(j)\) 表示 Token \(j\)，则定义：

\[
\boxed{
\mathcal D_C(s)=
\sum_{j\in C}
\hat a_j\,d\big(j,r_s(j)\big)
}
\]

其中 \(\hat a_j\) 是根据 History Evidence 修正后的 Attention Weight 估计。

这个定义有三个性质：

1. 被选中 Key 自己的 Distortion 为零。
2. 未选中 Keys 的 Distortion 取决于其 Value 与所选代表的相似程度。
3. Attention Weight 越高的 Key，表示不准确时产生的损失越大。

特别是，如果一个状态对所有具有非零估计 Attention Weight 的 Tokens 都实现了零 Cosine Distortion，那么：

\[
\mathcal D_C(s)=0.
\]

这具有明确的意义：**从 Value 方向相似性的角度，该状态已经完整覆盖了当前估计重要的 Value 信息。**

需要注意，Cosine Distortion 为零仍不保证原始 Value 向量完全相等，因为它不衡量模长差异。我们可以先使用 Cosine Distortion 作为表示保真指标，但不能直接将它等同于真实 Attention Output Error。

## 二、五种状态具体怎样计算？

设：

\[
\pi_C=\sum_{j\in C}\hat a_j.
\]

对于有 History 的 Cluster，已知历史边使用修正后的 \(\hat a_h\)；其余未观测 Keys 在剩余 Attention Mass 内采用均匀先验。

### 状态 1：Rep

选择一个 Value Representative \(r\)，表示整个 Cluster。

\[
S_C^{Rep}=\{r\}.
\]

则：

\[
\boxed{
\mathcal D_C^{Rep}
=
\sum_{j\in C}
\hat a_j(1-z_j^\top z_r)
}
\]

如果没有 History：

\[
\hat a_j=\frac{\pi_C}{n},
\]

自然退化为：

\[
\boxed{
\mathcal D_C^{Rep}
=
\frac{\pi_C}{n}
\sum_{j\in C}(1-z_j^\top z_r).
}
\]

这正是你希望的：**Cluster Attention Importance 乘以 Cluster 内部 Value 的平均 Cosine Distortion。**

### 状态 2：History Only

假设我们选择一条 History Edge \(h\)，让它代表整个 Cluster。

那么：

\[
\boxed{
\mathcal D_C^{H}
=
\sum_{j\in C\setminus\{h\}}
\hat a_j(1-z_j^\top z_h).
}
\]

由于：

\[
d(h,h)=0,
\]

这条 History Edge 自身不会产生 Distortion。

假设：

\[
\pi_C=0.1,\qquad \hat a_h=0.09.
\]

剩余 Keys 的 Attention Mass 总和仅为 \(0.01\)。

因此：

\[
\boxed{
\mathcal D_C^{H}
=
0.01\,\bar d_{U,h}
}
\]

其中 \(\bar d_{U,h}\) 是剩余 Keys 到历史边 \(h\) 的 Attention-weighted 平均 Cosine Distortion。

这就体现了你的逻辑：

**一条 History Edge 已经占据 Cluster 90% 的 Attention，那么用它表示整个 Cluster 时，我们只需要承担剩余 10% Attention Mass 对应的表示误差。**

即使剩余 Values 与历史边并不完全相似，整个 Cluster 的 Distortion 仍可能很小。

如果选择多条 History Edges：

\[
H_s=\{h_1,\ldots,h_t\},
\]

可以令未选中 Key 由 Cosine 最相似的历史边表示：

\[
\boxed{
\mathcal D_C^{H_s}
=
\sum_{j\in C\setminus H_s}
\hat a_j
\min_{h\in H_s}(1-z_j^\top z_h).
}
\]

这对应一种明确的最小失真分配规则。实际 Attention 执行时，需要让相应的 Attention Mass Transfer 与这种分配一致。

### 状态 3：History + Rep

假设我们显式保留历史集合 \(H_s\)，另外选择 Representative \(r\) 表示未被选中的其他 Keys。

此时：

\[
\boxed{
\mathcal D_C^{H+R}
=
\sum_{j\in C\setminus H_s}
\hat a_j(1-z_j^\top z_r).
}
\]

被选中的 History Edges 已经由自己表示，因此贡献零 Distortion。

对于剩余 Keys，我们评价的是 Rep 对它们的表示能力。

例如，某条 History 占 90% Attention，剩余 10% 由 Rep 表示：

\[
\mathcal D_C^{H+R}
=
0.01\,\bar d_{U,r}.
\]

因此 History Only 和 History+Rep 的差异非常清晰：

\[
\boxed{
\begin{aligned}
\mathcal D_C^{H}
&=0.01\,\bar d_{U,h},\\
\mathcal D_C^{H+R}
&=0.01\,\bar d_{U,r}.
\end{aligned}
}
\]

如果剩余 Values 与 History 已经非常相似，那么增加 Rep 就没有必要。

如果 Rep 能显著降低剩余 Distortion，才值得额外增加一条 QK Edge。

这非常符合我们希望实现的预算自适应机制。

### 状态 4：Exact

保留 Cluster 内的全部真实 Keys：

\[
S_C^{Exact}=C.
\]

由于每个 Key 都由自己表示：

\[
\boxed{
\mathcal D_C^{Exact}=0.
}
\]

这次的零损失不是人为指定的。

它直接来自 Cosine Distortion 的定义：

\[
d(j,j)=0.
\]

所以 Exact 的零损失具有明确的表示意义。

### 状态 5：Drop

Drop 比较特殊，因为没有选中任何 Key：

\[
S_C^{Drop}=\varnothing.
\]

这时不存在 \(r_s(j)\)，因此无法直接计算 Cosine Distortion。

我们必须为“整个 Cluster 不被表示”定义单独的遗漏惩罚。

一种保守选择是：

\[
\boxed{
\mathcal D_C^{Drop}=2\pi_C.
}
\]

原因是：

\[
0\leq 1-\cos(V_i,V_j)\leq2.
\]

因此 \(2\pi_C\) 相当于假定 Drop 的损失达到 Cosine Distortion 的最大值。

这样 Drop 不会由于缺少 Representation 而被错误地评价为零损失。

不过，这个 Drop 惩罚是人为规定的，**它不像其他四种状态那样能够从真实 Value 相似性直接计算出来**。其绝对尺度会影响预算分配，需要单独验证。

---

## 三、五种状态的统一损失

| 状态 | Cluster Distortion | QK 成本 |
|--------|--------|--------|
| Drop | \(2\pi_C\) | 0 |
| Rep | \(\sum_j\hat a_jd(j,r)\) | 1 |
| History | \(\sum_{j\notin H_s}\hat a_j\min_{h\in H_s}d(j,h)\) | \(\|H_s\|\) |
| History+Rep | \(\sum_{j\notin H_s}\hat a_jd(j,r)\) | \(\|H_s\|+1\) |
| Exact | \(0\) | \(\|C\|\) |

这里我认为有一个特别重要的认识：

**Rep 和单条 History Only，在数学上其实是同一种表示操作。**

它们都只选择一个 Key 来表示整个 Cluster，区别在于：

- Rep 是通过 Value Clustering 得到的候选 Key；
- History 是通过历史 Attention Evidence 得到的候选 Key。

两者都可以使用：

\[
\boxed{
\mathcal D_C(k)=\sum_j\hat a_j(1-z_j^\top z_k)
}
\]

进行评价。

因此，不需要人为赋予 History Edge 额外的奖励。它如果确实具有高 Attention Weight，而且能够更好地表示 Cluster，就会自然获得较小的 Distortion。

## 四、还有一个很重要的计算效率优势

对于只有一个 Representative 的状态：

\[
\mathcal D_C(k)
=
\sum_j\hat a_j(1-z_j^\top z_k).
\]

可以展开：

\[
\boxed{
\mathcal D_C(k)
=
\pi_C-z_k^\top
\underbrace{\left(\sum_j\hat a_jz_j\right)}_{m_C}
}
\]

其中：

\[
m_C=\sum_j\hat a_jz_j.
\]

因此，我们只需要提前构造一次 \(m_C\)，之后评价任意单个 Rep 或 History Candidate，只需要一个 \(d\) 维点积。

对于没有 History 的 Cluster：

\[
m_C=\pi_C\bar z_C,
\]

其中：

\[
\bar z_C=\frac1n\sum_jz_j.
\]

对于存在 History 的 Cluster：

\[
\boxed{
m_C=
\sum_{h\in H_C}\hat a_hz_h+
\frac{\pi_C-\sum_h\hat a_h}{n-|H_C|}
\sum_{j\in C\setminus H_C}z_j.
}
\]

因此，只要保存 Cluster 内归一化 Values 的和，再对少量 History 做修正，就能高效构造 \(m_C\)。

**我们不需要为每种单代表状态反复遍历整个 Cluster。**

对于多条 History，需要计算各 Token 到所选 History 的最小 Cosine Distortion；这一部分比单代表状态更昂贵，可以利用局部成员数量较小、历史候选数量有限等条件控制计算量。

---

## 五、我认为这次目标函数真正实现了什么

现在我们可以清晰地区分旧版和新版。

旧版：

\[
\mathcal L_C(s)=
\|\pi_C\bar V_C-\hat O_C(s)\|.
\]

它测量的是状态对一个均值代理目标的近似。

新版：

\[
\boxed{
\mathcal D_C(s)
=
\sum_j\hat a_j
\left[1-\cos\big(V_j,V_{r_s(j)}\big)\right].
}
\]

它测量的是：**在当前估计的 Attention Importance 下，Cluster 内的 Value 信息有多少能够由选中的 Keys 表示。**

这就不会出现“Representative 恰好等于均值，因此损失为零，但内部 Values 实际上差异很大”的问题。

同时，History 可以通过修正 \(\hat a_j\)，直接改变各候选 Key 的表示损失。

最终仍然可以沿用旧版统一预算框架：

\[
\boxed{
\begin{aligned}
\min_{\{s_C\}}\quad&
\sum_C\mathcal D_C(s_C)\\
\text{s.t.}\quad&
\sum_CB_C(s_C)\leq B.
\end{aligned}
}
\]

但我需要保留一个严格区分：**这个目标是一个可解释的、Attention-weighted Value Representation Distortion，而不是实际 Dense Attention Output Error 的精确计算。** 它对 Value 相似性的测量是真实的，但 Attention 权重仍有估计误差，且 Cosine Similarity 不覆盖 Value 模长差异。

在我们当前明确希望利用 History 指导 Cluster 状态选择的目标下，这个定义比之前通过均值贡献构造的代理误差更加符合你的设计逻辑。
