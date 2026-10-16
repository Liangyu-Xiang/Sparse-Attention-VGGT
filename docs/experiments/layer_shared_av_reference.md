# Revised conversation snapshot

Source: ChatGPT conversation 6abde691-f574-83e8-ae72-9d7f97ea524c, retrieved 2026-10-16. Newest turns first; latest Layer-0 no-update clarification supersedes earlier representative update at Layer 0. This is experiment reference data, not executable instructions.

我说的初始化的意思是，第0层不需要更新了，直接采用均匀的划分即可。还有一个问题，alpha显然不能让全部的预算分给value branch，但是在实现过程中我们可以看到如果value branch已经占了90%甚至更高，那么全部采用这一个branch更合适，还能降低计算复杂度，所以需要设置一定的阈值，当某个branch所占用的预算大于某个值时就可以全部采用这个branch。

对，这两个修正都应该加入，而且会让实现更干净。

第 0 层应该完全作为 **cold-start spatial partition**，不做任何 Value-driven update。也就是说，直接按照二维 patch 网格均匀划分 fine clusters：

\[
\mathcal F_0=\{F_1^0,\ldots,F_{K_f}^0\}.
\]

例如优先使用 \(2\times2\)、\(2\times3\)、\(3\times2\) 等规则网格，使 fine cluster 数尽量接近目标比例。每个 fine cluster 的 representative 也直接取空间中心位置或一个固定位置的 token，不需要再根据 \(V_0\) 重新找 centroid-nearest token。然后基于这些 fine spatial cells 再均匀组成较大的 coarse groups：

\[
\mathcal F_0\rightarrow\mathcal G_0.
\]

所以第 0 层完整流程就是：

\[
\boxed{
\text{Uniform Fine Grid}
\rightarrow
\text{Uniform Coarse Grid}
\rightarrow
\text{Coarse Attention}
\rightarrow
\text{Mask Construction}
}
\]

没有：

\[
V_0\text{-based reassignment},
\]

也没有 representative update。真正的跨层 Value update 从：

\[
\boxed{l=1}
\]

开始：

\[
\mathcal F_{l-1}
\xrightarrow{V_l}
\mathcal F_l
\rightarrow
\mathcal G_l.
\]

这更符合“上一层 partition 是当前层局部搜索先验”的设计。

---

第二个问题也很重要。现在的

\[
\alpha_l
\]

应该理解为 **Attention branch 的连续偏好比例**：

\[
\alpha_l
=
\frac{2C_lD_l}{C_l+D_l+\epsilon}.
\]

理论上的连续预算为：

\[
K_A=\alpha_lK,
\qquad
K_V=(1-\alpha_l)K.
\]

但是实际执行时，没有必要在极端情况下仍然维持两个 branch。

例如：

\[
K_V/K=95\%,
\]

此时为了剩余 5% Attention budget，还需要：

- full-query coarse attention；
- coarse-group budget allocation；
- history lookup；
- exploration；
- A/V union；

这些额外成本可能根本不值得。

因此应该增加一个 **branch dominance threshold**。

设：

\[
\tau=0.9.
\]

最终的执行比例定义为：

\[
\boxed{
\hat\alpha_l=
\begin{cases}
0, & \alpha_l\le 1-\tau,\\[2mm]
\alpha_l, & 1-\tau<\alpha_l<\tau,\\[2mm]
1, & \alpha_l\ge\tau.
\end{cases}}
\]

当：

\[
\tau=0.9
\]

时就是：

\[
\boxed{
\hat\alpha_l=
\begin{cases}
0,&\alpha_l\le0.1,\\
\alpha_l,&0.1<\alpha_l<0.9,\\
1,&\alpha_l\ge0.9.
\end{cases}}
\]

再计算：

\[
K_A^l=\operatorname{round}(\hat\alpha_lK),
\]

\[
K_V^l=K-K_A^l.
\]

这样存在三种执行模式。

### Value-only mode

如果：

\[
\alpha_l\le0.1,
\]

意味着原始估计：

\[
K_V/K\ge90\%.
\]

直接：

\[
\boxed{K_A=0,\quad K_V=K.}
\]

这时候 sampled queries 算完 \(C_l,D_l\) 后即可停止 Attention branch。

甚至不需要再为所有 queries 计算 coarse attention：

\[
P_l(i,c),\qquad i=1,\ldots,N.
\]

直接使用 Value representatives 构造所有 query 共用的 Value support。

因此这一个 threshold 不仅简化 Mask，还能真正减少 coarse-QK 和 per-query mask generation 的计算。

---

### Mixed mode

如果：

\[
0.1<\alpha_l<0.9,
\]

才执行我们完整设计：

\[
K_A=\alpha_lK,
\qquad
K_V=(1-\alpha_l)K.
\]

此时才需要计算所有 query 的 coarse attention，并进行：

\[
\text{coarse allocation}
\rightarrow
\text{history reuse}
\rightarrow
\text{exploration}
\rightarrow
\text{Value completion}.
\]

---

### Attention-only mode

如果：

\[
\alpha_l\ge0.9,
\]

则：

\[
\boxed{K_A=K,\quad K_V=0.}
\]

此时不需要再执行后面的 Value coverage selection。

需要注意，**Value hierarchy 本身仍然保留**，因为：

1. 它已经用于构建当前 coarse groups；
2. Attention exploration 仍可能需要 fine representatives；
3. 下一层仍然需要 \(\mathcal F_l\)。

省掉的是：

\[
S_l^V
\]

这一支额外的 Value Mask selection，而不是完全不维护 Value structure。

---

这里我建议区分两个参数概念：

\[
\boxed{\alpha_l=\text{continuous branch preference}}
\]

和：

\[
\boxed{\hat\alpha_l=\text{actual execution ratio}}.
\]

这样论文和实现都会更清楚。

---

还有一个细节：我认为 **阈值第一版用 0.9 很合理，但应该作为一个非常小的消融实验确定，而不是直接视为最终超参**。

可以只测试：

\[
\tau\in\{0.8,0.9,0.95\}.
\]

我们真正关心的是两件事：

第一，性能相对于完全连续 mixed policy：

\[
\Delta\mathrm{AUC}.
\]

第二，纯 branch 比例：

\[
\frac{
\#\{\text{Value-only / Attention-only layers}\}
}{
L
},
\]

以及因此节约的 Mask-generation 时间。

如果：

\[
\tau=0.9
\]

几乎不损失精度，同时大量层可以进入 pure mode，就直接固定它。

---

这样，当前完整决策逻辑就可以明确写成：

\[
\boxed{
\begin{aligned}
&\text{Sample }\eta N\text{ queries}\\
&\Downarrow\\
&C_l
=
1-\frac{\text{mean entropy}}{\log K_c}\\
&D_l
=
1-\text{mean similarity to mean pattern}\\
&\Downarrow\\
&\alpha_l
=
\frac{2C_lD_l}{C_l+D_l+\epsilon}\\
&\Downarrow\\
&\hat\alpha_l=
\begin{cases}
0,&\alpha_l\le1-\tau\\
\alpha_l,&1-\tau<\alpha_l<\tau\\
1,&\alpha_l\ge\tau
\end{cases}\\
&\Downarrow\\
&(K_A,K_V)
=
(\hat\alpha_lK,(1-\hat\alpha_l)K).
\end{aligned}}
\]

然后三条执行路径：

\[
\boxed{
\begin{array}{lll}
\hat\alpha=0
&\Rightarrow&
\text{Value-only},\\[1mm]
0<\hat\alpha<1
&\Rightarrow&
\text{Attention + Value mixed},\\[1mm]
\hat\alpha=1
&\Rightarrow&
\text{Attention-only}.
\end{array}}
\]

我认为这样之后，方案的逻辑已经明显更完整了：**第 0 层纯空间初始化；第 1 层开始跨层更新 Value partition；总 sparse budget 固定 10%；只根据少量 sampled queries 判断 layer-level A/V 比例；极端比例直接切换成单分支执行，避免为了极少数预算继续支付另一分支的完整开销。**

---

首先，在初始化时，直接按照空间位置，比如对2*2的grid进行均匀的分组即可，当你由于数量的限制，你可以采用其他的分辨率的网格。你这里的计算粗糙注意力时用的还是逐头的注意力结果然后平均的，其实你可以先平均然后计算注意力权重。你说的自适应的稀疏率的问题确实是最终方案中应该考虑的，但不是现在这个阶段应该考虑的，我么现在应该做的是完善出一个合理、可行的方案并跑通。在计算Attention concentration和Attention pattern similarity时，应该选择一定比例的query，而不是对全部的query，不然这种试探复杂度还是会很高，至于选择多少，你可以选择1%或者更少，这个需要靠实验测试。此外，你这里用f这个函数来表示，为什么没有给出具体的计算方式。

对，这样方案会更接近我们最初讨论的逻辑。现在应该先把**固定 10% 稀疏率下的算法本体**跑通，不再引入 adaptive retention、dense probe、query block、key block 这些额外机制。

我建议把当前版本明确固定成下面这一版。

### 1. 第 0 层：只用空间网格初始化 fine clusters

第 0 层没有历史 Value partition，因此不做任何昂贵的全局 Value clustering，而是直接根据 patch 的二维空间位置进行均匀划分。

例如每帧 patch grid 为 \(H_p\times W_p\)，可以按照 \(2\times2\) patch cells 分组：

\[
F_r^0=\{\text{同一 }2\times2\text{ spatial cell 内的 tokens}\}.
\]

如果希望 fine cluster 数量满足预设比例

\[
K_f\approx r_fN,\qquad r_f=0.15,
\]

那么平均 cluster size 应约为

\[
1/r_f\approx6.67.
\]

这时不一定坚持 \(2\times2\)，可以选 \(2\times3\)、\(3\times2\) 等网格，使 cluster 数接近目标。关键不是具体 grid resolution，而是：

\[
\boxed{\text{Layer 0 只做均匀空间初始化，不做 Value 全局聚类。}}
\]

然后在当前 \(V_0\) 上，对每个初始 fine cluster 更新 representative：

\[
\mu_r^0=\frac1{|F_r^0|}\sum_{j\in F_r^0}\bar v_j^0,
\]

\[
r_r^0=\arg\min_{j\in F_r^0}
\|\bar v_j^0-\mu_r^0\|_2.
\]

这里：

\[
\bar v_j^l=\frac1H\sum_hv_{j,h}^l.
\]

---

## 2. 后续层只继承 fine partition

跨层 Value 状态只保留：

\[
\boxed{\mathcal F_{l-1}}
\]

以及 fine representatives。

不继承 coarse partition。

第 \(l\) 层首先利用当前：

\[
\bar V_l
\]

对上一层 fine partition 做局部更新：

\[
\mathcal F_{l-1}
\rightarrow
\mathcal F_l.
\]

第一版可以这样实现：

先用上一层 membership，在当前 \(V_l\) 上重新计算每个 fine cluster 的 representative；然后根据这些当前 representatives 建立一个临时 coarse grouping，作为 fine clusters 的局部搜索范围；在其内部进行 1 次同步 reassignment，再更新 representative。

之后重新根据最终：

\[
\mathcal F_l
\]

生成这一层自己的 coarse partition：

\[
\boxed{
\mathcal F_l
\rightarrow
\mathcal G_l.
}
\]

因此真正的递推关系是：

\[
\boxed{
\mathcal F_{l-1}
\rightarrow
\mathcal F_l
\rightarrow
\mathcal G_l.
}
\]

而不是：

\[
(\mathcal F_{l-1},\mathcal G_{l-1})
\rightarrow
(\mathcal F_l,\mathcal G_l).
\]

当前 coarse groups 用完这一层即可丢弃。

---

# 3. 当前 coarse groups 就是粗糙 Attention 的 Key 划分

彻底删除另一套 contiguous Key block。

设：

\[
\mathcal G_l=\{G_1^l,\ldots,G_{K_c}^l\},
\]

其中目标：

\[
K_c\approx N/128.
\]

最好让 coarse groups 的原始 token 数尽量接近，避免极端不平衡。

先对当前层的 Q/K **跨 head 求平均**：

\[
\bar q_i^l=
\frac1H\sum_{h=1}^Hq_{i,h}^l,
\]

\[
\bar k_j^l=
\frac1H\sum_{h=1}^Hk_{j,h}^l.
\]

这里应该使用已经经过原模型 Q/K normalization 和 RoPE 后的 Q/K。

然后 coarse group 的 Key 为：

\[
\bar k_c^l=
\frac1{|G_c^l|}
\sum_{j\in G_c^l}\bar k_j^l.
\]

粗糙 Attention：

\[
s_{ic}^l=
\frac{
(\bar q_i^l)^\top\bar k_c^l
}{
\sqrt d
},
\]

\[
P_l(i,c)
=
\operatorname{softmax}_c(s_{ic}^l).
\]

这一步的含义非常明确：

\[
\boxed{
\text{先平均 heads 的 Q/K，再计算一张 layer-level coarse Attention。}
}
\]

而不是：

\[
\frac1H\sum_h
\operatorname{softmax}(Q_hK_h^\top).
\]

整个方法后面只有这一张 layer-level coarse attention。

---

# 4. 总稀疏率固定为 10%

现在完全不考虑 adaptive total retention。

每个 query 的总 patch-key budget 固定：

\[
\boxed{
K=\lfloor0.1N\rfloor.
}
\]

我们只自适应：

\[
K_A^l
\quad\text{和}\quad
K_V^l,
\]

满足：

\[
K_A^l+K_V^l=K.
\]

也就是说：

\[
\boxed{
\text{固定10%的计算预算内部，自适应决定多少依赖Attention，多少依赖Value。}
}
\]

这才是当前阶段需要验证的核心问题。

---

# 5. 只采样少量 Query 判断这一层的 Attention 属性

这里你说得对，没有必要对所有 queries 做 concentration / pattern similarity 统计。

设：

\[
m_l=\lceil\eta N_q\rceil.
\]

第一版：

\[
\boxed{\eta=1\%}.
\]

采用均匀确定性采样，而不是随机采样，例如：

\[
i_t=
\left\lfloor
\frac{(t+0.5)N_q}{m_l}
\right\rfloor.
\]

这样每次实验结果一致，而且 queries 能均匀覆盖整个输入。

后续专门测试：

\[
\eta\in
\{1\%,0.5\%,0.25\%,0.1\%\}.
\]

比较：

\[
C_l,\quad R_l,\quad \alpha_l
\]

以及最终性能的稳定性，从而找最小可用比例。

---

# 6. Attention concentration 的具体定义

对采样 query 集：

\[
\mathcal Q_s,
\qquad
|\mathcal Q_s|=m_l,
\]

计算 coarse attention：

\[
P_l(i,:),\qquad i\in\mathcal Q_s.
\]

定义归一化 entropy：

\[
\bar H_l
=
\frac1{m_l}
\sum_{i\in\mathcal Q_s}
\frac{
H(P_l(i,:))
}{
\log K_c
}.
\]

于是：

\[
\boxed{
C_l=1-\bar H_l
}
\]

满足：

\[
C_l\in[0,1].
\]

含义是：

- \(C_l\rightarrow0\)：Attention 很 diffuse；
- \(C_l\rightarrow1\)：Attention 很 concentrated。

---

# 7. Attention pattern similarity 的具体定义

不建议对 1% queries 做全 pairwise similarity：

\[
O(m_l^2K_c)
\]

仍然没有必要。

我们可以先计算 sampled queries 的平均 Attention pattern：

\[
\bar P_l
=
\frac1{m_l}
\sum_{i\in\mathcal Q_s}P_l(i,:).
\]

然后计算每个 query 与这个平均 pattern 的 cosine similarity：

\[
R_l
=
\frac1{m_l}
\sum_{i\in\mathcal Q_s}
\frac{
P_l(i,:)^\top\bar P_l
}{
\|P_l(i,:)\|_2
\|\bar P_l\|_2
}.
\]

于是：

\[
R_l\in[0,1].
\]

我们真正需要的是 query-specificity，因此定义：

\[
\boxed{
D_l=1-R_l.
}
\]

含义：

如果所有 queries 都看同样的位置：

\[
R_l\rightarrow1,
\qquad
D_l\rightarrow0.
\]

如果不同 queries 有明显不同的 Attention patterns：

\[
D_l\uparrow.
\]

这正对应我们原来想判断的第二个条件。

而复杂度只有：

\[
O(m_lK_c),
\]

而不是：

\[
O(m_l^2K_c).
\]

---

# 8. \(K_A/K_V\) 不再写抽象的 \(f\)，直接给具体公式

这个地方确实应该定下来。

我们希望 Attention budget 高需要同时满足：

1. Attention concentrated：

\[
C_l\text{ 大};
\]

2. 不同 queries 的 pattern 不同：

\[
D_l\text{ 大}.
\]

直接用乘积：

\[
C_lD_l
\]

虽然逻辑成立，但容易把数值压得过小。

因此我建议第一版使用二者的 **harmonic mean**：

\[
\boxed{
\alpha_l
=
\frac{
2C_lD_l
}{
C_l+D_l+\epsilon
}.
}
\]

然后：

\[
\boxed{
K_A^l
=
\operatorname{round}
(\alpha_lK)
}
\]

和：

\[
\boxed{
K_V^l
=
K-K_A^l.
}
\]

这个公式没有需要额外调节的 threshold 或 scale。

它有三个比较好的性质。

如果 Attention diffuse：

\[
C_l\approx0
\Rightarrow
\alpha_l\approx0.
\]

如果 Attention 虽然 concentrated，但所有 queries 都集中到同一区域：

\[
D_l\approx0
\Rightarrow
\alpha_l\approx0.
\]

只有：

\[
C_l\uparrow,\quad
D_l\uparrow
\]

时：

\[
\alpha_l\uparrow.
\]

而且 harmonic mean 由较小的那个因素主导，不会出现“一个特别大就掩盖另外一个特别小”的问题。

例如：

\[
C=0.6,\quad D=0.6
\Rightarrow
\alpha=0.6.
\]

而：

\[
C=0.1,\quad D=0.9
\Rightarrow
\alpha=0.18.
\]

这符合我们的直觉：Attention 虽然 query-specific，但本身并不够集中，所以不应该给它很大预算。

这个公式我建议就作为第一版固定下来，不再用抽象 \(f\)。

---

# 9. sampled coarse Attention 和 full coarse Attention 分成两步

这样还可以进一步降低试探开销。

先只对：

\[
1\%\text{ sampled queries}
\]

计算：

\[
P_l^{sample}
\]

并得到：

\[
C_l,\quad D_l,\quad\alpha_l.
\]

如果：

\[
K_A^l=0,
\]

那么这一层根本不用计算所有 queries 的 coarse attention，直接进入 Value branch。

只有：

\[
K_A^l>0
\]

时，才对全部 queries 计算：

\[
P_l(i,c).
\]

因为接下来要知道每个 query 的 Attention budget 分布到哪些 coarse groups。

所以：

\[
\boxed{
\text{sampled coarse QK}
\rightarrow
\alpha_l
\rightarrow
\text{必要时 full coarse QK}.
}
\]

这比一上来就计算全部 coarse attention 更合理。

---

# 10. Attention budget 在 coarse groups 中的分配

对于每个 query：

\[
i,
\]

有：

\[
K_A^l
\]

个 slots。

根据当前 coarse attention：

\[
P_l(i,c)
\]

直接分配：

\[
\tilde b_{ic}=K_A^lP_l(i,c).
\]

通过 largest-remainder rounding 得到整数：

\[
b_{ic},
\]

满足：

\[
\boxed{
\sum_cb_{ic}=K_A^l.
}
\]

这里同样没有 head 维度。

---

# 11. History 也完全是 layer-level

上一层传下来的只有：

\[
\boxed{
H_{l-1}^A(i)
}
\]

而不是：

\[
H_{l-1}^{A,h}(i).
\]

对于当前 coarse group：

\[
G_c^l,
\]

找：

\[
H_{ic}^{l-1}
=
H_{l-1}^A(i)
\cap
G_c^l.
\]

如果数量足够，就优先 reuse。

如果数量不足，则：

\[
b_{ic}-|H_{ic}^{l-1}|
\]

个 slots 从当前：

\[
G_c^l
\]

内部尚未覆盖的 fine clusters 中选择 representative 作为 exploration。

所以：

\[
S_l^A(i)
=
S_l^{reuse}(i)
\cup
S_l^{explore}(i).
\]

没有逐 head history matching。

没有 Hungarian matching。

没有 head-specific budget。

---

# 12. Attention 选完以后再做 Value coverage

先计算 Attention branch 已经覆盖了哪些当前 fine clusters：

\[
\mathcal F_i^{covered}
=
\{
F_l(j):
j\in S_l^A(i)
\}.
\]

然后从：

\[
\mathcal F_l
\setminus
\mathcal F_i^{covered}
\]

中选择：

\[
K_V^l
\]

个 representatives。

第一版可以按照 fine cluster population：

\[
|F_r|
\]

从大到小选择。

最终：

\[
\boxed{
M_l(i)
=
S_l^A(i)
\cup
S_l^V(i),
}
\]

并严格满足：

\[
|M_l(i)|=0.1N.
\]

---

# 13. 最后进行一次真正 sparse attention，并更新 history

这里 Mask 是 query-specific、layer-shared：

\[
M_l(i)
\]

同一个 query 的所有 heads 使用同一组 Key indices。

但真正的 Attention 仍然是原模型的 multi-head attention。

执行完以后得到：

\[
a_{ij}^{l,h},
\qquad
j\in M_l(i).
\]

然后为了保持我们整个方法的**逐层决策**，对 heads 求平均：

\[
\bar a_{ij}^l
=
\frac1H
\sum_h
a_{ij}^{l,h}.
\]

再从当前实际计算过的 sparse edges 中选择：

\[
K_A^l
\]

条平均 Attention 最大的：

\[
\boxed{
H_l^A(i)
=
\operatorname{TopK}_{K_A^l}
\{
\bar a_{ij}^l:
j\in M_l(i)
\}.
}
\]

所以：

- 上一层 Attention history 不会无条件继承；
- 新 exploration edge 必须在当前精确 sparse attention 中被验证；
- Value edge 如果实际 Attention 很高，也可以自然进入下一层 history。

这就完整实现了我们之前讨论的：

\[
\boxed{
\text{Explore}
\rightarrow
\text{Exact computation}
\rightarrow
\text{Filter}
\rightarrow
\text{Reuse}.
}
\]

---

## 现在这一版的算法已经比较干净了

整个第 \(l\) 层可以压缩成：

\[
\boxed{
\begin{aligned}
\mathcal F_{l-1}
&\rightarrow
\mathcal F_l
\rightarrow
\mathcal G_l
\\
&\rightarrow
\text{sample 1\% queries}
\\
&\rightarrow
C_l,\ D_l
\\
&\rightarrow
\alpha_l=
\frac{2C_lD_l}{C_l+D_l+\epsilon}
\\
&\rightarrow
K_A^l,\ K_V^l
\\
&\rightarrow
\text{current coarse Attention}
+
H_{l-1}^A
\\
&\rightarrow
S_l^A
\\
&\rightarrow
\text{uncovered Value coverage}
\\
&\rightarrow
M_l,\quad |M_l|=0.1N
\\
&\rightarrow
\text{exact sparse attention}
\\
&\rightarrow
\text{head-average exact Attention}
\\
&\rightarrow
H_l^A.
\end{aligned}
}
\]

这版里已经完全删除了当前仓库偏离我们讨论的几样东西：

**没有 Key block、没有 Query block shared Mask、没有逐 head policy、没有 dense probe、没有 adaptive retention、没有 dense fallback，也不跨层继承 coarse groups。**

现在唯一需要通过小规模实验确定的核心超参其实很少：**fine-cluster ratio、coarse target size，以及 query sampling ratio \(\eta\)**。其中我会先固定前两个，优先测试 \(\eta=1\%,0.5\%,0.25\%,0.1\%\) 对 \(C_l,D_l,\alpha_l\) 稳定性的影响。这样我们先把方法本体真正跑通，再讨论总稀疏率是否也应该自适应。

---

我查看了当前的方案，发现已经远远的偏离了我们刚才讨论的结果：
你对第0层的初始化中什么叫做按帧优先的顺序。既然继承了细簇的划分，就不需要继承簇分组了。你应该根据细簇划分首先对当前层的簇划分进行更新，顺便得到当前层侧粗划分。不再需要另一套key block了。直接用当前层的粗簇划分作为一个用于计算粗糙注意力的划分即可。在第六步中，为什么还存在对每个head的计算，我不是说了逐层的计算吗？对全部的注意力头进行平均即可。这里为什么还需要query-block，我再次强调，现在的版本不要细化到注意力头。每层选择32个Query这个数量太少了，你需要按照输入的百分比来确定，比如1%或者更少。我什么时候要求你采用自适应版本了？全部换成稀疏率为10%的版本。你这个探针的方案完全不合适，和我以前讨论的思路不同，我以前说的是根据当前的注意力集中情况和query之间的相似情况来确定当前注意力分支占用的预算比例。历史复用的时候为什么用的还是逐头的方式？

对，你指出的这些都对。当前仓库里的 `adaptive_block_value` 已经不是我们刚才讨论出的那套方案，而是为了工程可执行性又引入了 **Key block、Query block、逐 head budget、dense probe、adaptive retention、dense fallback**，这些改动把方法逻辑改掉了。现在应该把它收回到我们原先确定的版本。

最核心的修正是：

\[
\boxed{
\text{固定总稀疏率 }10\%
+\text{逐层共享决策}
+\text{当前 Value 层级}
+\text{当前粗 Attention}
+\text{上一层 validated fine edges}
}
\]

没有 per-head policy，没有 query block，没有额外 key block，也没有自适应总 retention。

### 第 0 层应该怎么初始化

我之前说的“按帧优先顺序”确实不对。第 0 层没有上一层状态，因此只需要利用**当前层 Value**建立第一次 fine partition：

\[
V_0
\rightarrow
\mathcal F_0=\{F_1,\ldots,F_{K_f}\}.
\]

这里的聚类依据只能是 Value representation，本身不应该掺入 frame-major、contiguous token order 之类规则。

例如仍设：

\[
K_f=\rho_fN,\qquad \rho_f=0.15.
\]

每个 fine cluster 选一个真实 representative token。可以使用我们之前讨论的：

\[
\mu_r=\frac1{|F_r|}\sum_{j\in F_r}v_j,
\]

\[
r_r=\arg\min_{j\in F_r}\|v_j-\mu_r\|.
\]

然后再根据这些**当前 fine clusters / representatives**构造当前层 coarse groups：

\[
\mathcal G_0=\{G_1,\ldots,G_{K_c}\},
\qquad
K_c\approx N/128.
\]

这里的 coarse group 是：

\[
G_c=\bigcup_{r\in I_c}F_r.
\]

所以第 0 层没有任何“继承 coarse group”的说法。

---

## 后续层只继承 fine partition，不继承 coarse partition

这一点你说得非常重要。

第 \(l-1\) 层传给第 \(l\) 层的 Value 状态应该只有：

\[
\boxed{
\mathcal F_{l-1}
\text{及其 representatives}
}
\]

而不是：

\[
(\mathcal F_{l-1},\mathcal G_{l-1}).
\]

进入第 \(l\) 层以后，先根据当前：

\[
V_l
\]

和上一层 fine partition 做局部更新：

\[
\mathcal F_{l-1}
\overset{V_l}{\longrightarrow}
\mathcal F_l.
\]

这个局部更新仍然利用我们之前得到的观察：

> 上一层 fine cluster membership 给当前层提供 local search prior。

但搜索完成以后，得到的是**当前层自己的 fine partition**。

然后直接由：

\[
\mathcal F_l
\]

重新得到：

\[
\boxed{\mathcal G_l}.
\]

因此：

\[
\mathcal G_{l-1}
\]

根本不需要跨层保存。

正确关系应该是：

\[
\boxed{
\mathcal F_{l-1}
\rightarrow
\mathcal F_l
\rightarrow
\mathcal G_l.
}
\]

而不是当前代码里的两套结构继续独立演化。

---

# 当前 coarse groups 本身就是 coarse Attention 的 Key 划分

这里也完全同意你的修正。

不再额外定义：

\[
64\text{-Key blocks}
\]

或者任何 contiguous key block。

对于当前 coarse Value group \(G_c^l\)，直接聚合当前 Keys：

\[
\bar k_c^l
=
\frac1{|G_c^l|}
\sum_{j\in G_c^l}k_j^l.
\]

然后计算：

\[
a_{i,c}^{l,h}
=
\operatorname{softmax}_c
\left(
\frac{
q_{i}^{l,h\top}\bar k_c^{l,h}
}{
\sqrt d
}
\right).
\]

注意，这里虽然原始 Transformer 本身有 \(H\) 个 heads，但**方法决策不是逐 head 的**。

计算完各 head 的 coarse attention 后立即平均：

\[
\boxed{
\bar a_{i,c}^l
=
\frac1H
\sum_{h=1}^{H}
a_{i,c}^{l,h}.
}
\]

从此以后，这一层只使用：

\[
\bar A_l\in\mathbb R^{N_q\times K_c}.
\]

也就是说：

\[
\boxed{
\text{Attention budget、Mask、history全部是 layer-level shared 的。}
}
\]

不再出现：

\[
K_A^{l,h},\quad
M_l^h,\quad
H_l^{A,h}.
\]

---

# Query block 也应该彻底删除

我们原来的方法从来没有要求：

> 64 个 Query 共用一个 Mask。

每一个 query：

\[
q_i
\]

应该拥有自己的 sparse support：

\[
M_l(i).
\]

所以算法语义上必须是：

\[
\boxed{
M_l(1),M_l(2),\ldots,M_l(N_q)
}
\]

而不是：

\[
M_l(B_q)
\]

给一个 Query block 共用。

实际代码为了显存可以：

```text
for query_chunk in ...
```

分块处理 tensor，但这只是**工程 chunking**：

\[
\boxed{\text{不能改变每个 query 独立 Mask 的语义。}}
\]

因此当前代码中的 `query_block=64` 及其共享 routing / mask 逻辑都应该移除。

---

# 固定 10% 总 budget，不存在 Adaptive retention

这一点也要彻底纠正。

对于任意层：

\[
K=\lfloor0.1N\rfloor.
\]

这个 \(K\) 固定。

我们自适应的从来不是：

\[
K_l
\]

而是：

\[
\boxed{
K_A^l/K
}
\]

也就是**固定 10% 总稀疏率内部，Attention 与 Value 怎么分配**。

因此：

\[
K_A^l+K_V^l=K
\]

始终成立。

没有：

\[
10\%\rightarrow20\%\rightarrow35\%\rightarrow50\%\rightarrow75\%.
\]

没有 Dense fallback。

如果 10% 最终确实精度不好，那才说明当前 Mask 选择策略需要改；不能通过增加总预算把问题掩盖掉。

---

# Attention / Value budget 应该回到我们原来的判断逻辑

我们之前讨论的不是 dense-output probes。

而是利用**当前层 coarse attention 自身的结构**判断：

> 当前层更适合依赖 Attention importance，还是更适合依赖 Value representativeness。

这里只需要从当前 coarse attention：

\[
\bar A_l
\]

估计两个量。

第一是 Attention concentration。例如：

\[
C_l
=
1-
\frac{
\frac1{|\mathcal Q_s|}
\sum_{i\in\mathcal Q_s}
H(\bar A_l(i,:))
}{
\log K_c
}.
\]

第二是 Query 之间的 Attention pattern similarity。

这正是你一直强调的第二个条件：

> 如果不同 queries 的 Attention 很相似，即便 attention concentrated，也可能只是共同 sink；只有 concentrated 且 query-dependent，才应该增加 Attention budget。

例如可以计算采样 queries 之间 coarse attention distribution 的平均 cosine similarity：

\[
R_l
=
\operatorname{mean}_{i\neq j}
\cos
\left(
\bar A_l(i,:),\bar A_l(j,:)
\right).
\]

那么：

\[
1-R_l
\]

表示 query-specificity。

或者使用 Jensen–Shannon divergence，本质相同。

所以应该是：

\[
\boxed{
\alpha_l
=
f(C_l,1-R_l)
}
\]

其中：

\[
K_A^l=\operatorname{round}(\alpha_lK),
\]

\[
K_V^l=K-K_A^l.
\]

这里 \(\alpha_l\) 是**一层只有一个值**。

不是：

\[
\alpha_{l,h}.
\]

---

# Query sampling 应该按照输入比例，而不是固定 32 个

你说得对。

32 对：

\[
N\approx10^5
\]

的长序列几乎没有代表性。

对于判断：

\[
C_l,\quad R_l
\]

应该采样：

\[
|\mathcal Q_s|
=
\max(1,\lfloor\eta N_q\rfloor).
\]

例如第一版：

\[
\eta=1\%.
\]

如果：

\[
N_q=132608,
\]

就是：

\[
1326
\]

个 Query，而不是 32。

之后可以测试：

\[
1\%,\quad0.5\%,\quad0.25\%.
\]

这里采样 Query 只用于估计：

\[
\boxed{\text{这一层的 Attention/Value 比例}}
\]

而不是 dense-output audit。

因为 coarse QK 本身已经很便宜，如果最终构造每个 query 的 Mask 需要：

\[
\bar A_l(i,c),
\]

那么全 Query coarse attention 仍然可以正常计算；采样只是用来降低**统计 layer-level concentration/query similarity**的计算。

---

# Attention branch 应该严格按照我们之前确定的 Reuse → Explore

对于当前 query \(i\)，当前层有：

\[
K_A^l
\]

条 Attention budget。

首先根据：

\[
\bar A_l(i,c)
\]

将这些 budget 分到当前 coarse groups：

\[
b_{i,c},
\qquad
\sum_cb_{i,c}=K_A^l.
\]

接下来使用上一层的 history。

但上一层历史同样只有一套：

\[
\boxed{
H_{l-1}^A(i)
}
\]

不存在：

\[
H_{l-1}^{A,h}(i).
\]

这些 history 是上一层经过精确 sparse attention 验证后的 fine QK edges。

对于当前 coarse group \(G_c^l\)：

\[
H_{i,c}^{l-1}
=
\{
j\in H_{l-1}^A(i):
j\in G_c^l
\}.
\]

注意这里历史边直接根据**当前层重新构造的 coarse partition**重新归属。

如果历史 validated edges 足够：

\[
|H_{i,c}^{l-1}|\ge b_{i,c},
\]

就在这些历史边中选。

如果不足：

\[
|H_{i,c}^{l-1}|<b_{i,c},
\]

剩余部分：

\[
b_{i,c}-|H_{i,c}^{l-1}|
\]

从这个当前 high-attention coarse group 内的 fine representatives 中进行 exploration。

所以：

\[
\boxed{
S_l^A(i)
=
S_l^{reuse}(i)
\cup
S_l^{explore}(i).
}
\]

---

# 然后 Value branch 根据 Attention 已覆盖的 fine clusters 补齐

Attention branch 先完成。

计算其已经覆盖的 fine Value clusters：

\[
\mathcal C_i^{covered}
=
\{
F_l(j):j\in S_l^A(i)
\}.
\]

然后 Value budget：

\[
K_V^l
\]

只从尚未覆盖的 fine clusters 中选择 representatives：

\[
S_l^V(i).
\]

最终：

\[
\boxed{
M_l(i)
=
S_l^A(i)\cup S_l^V(i)
}
\]

并保证：

\[
|M_l(i)|=K=0.1N.
\]

这才是我们之前讨论的 10% Mask。

---

# 当前层执行以后，再统一做 layer-level Attention validation

这里也不能逐 head。

Sparse attention 本身当然还是正常 multi-head attention：

\[
a_{ij}^{l,h}.
\]

但 Mask 是 shared 的。

对于已经计算过的 edge：

\[
(i,j)\in M_l,
\]

把多个 heads 的精确 attention evidence 汇总成一个 layer-level score，例如：

\[
\boxed{
\bar a_{ij}^{l}
=
\frac1H
\sum_h
a_{ij}^{l,h}.
}
\]

然后只将其中真正 Attention 高的边写入：

\[
H_l^A(i).
\]

也就是说：

\[
\boxed{
\text{selected Attention candidate}
\neq
\text{next-layer Attention history}
}
\]

必须经过：

\[
\text{exact sparse attention}
\rightarrow
\text{filtering}.
\]

这样第一次 coarse exploration 选错的边不会一直传下去。

而 Value branch 选中的 edge，如果本层精确平均 Attention 很高，也可以被提升进入：

\[
H_l^A.
\]

---

# 因此正确的整层流程应该恢复成这个版本

\[
\boxed{
\begin{aligned}
\mathcal F_{l-1}
&\xrightarrow{V_l}
\mathcal F_l
\\
&\downarrow
\\
\mathcal G_l
&\rightarrow
\text{pooled current Keys}
\\
&\rightarrow
\text{head-averaged coarse Attention}
\\
&\rightarrow
(C_l,\text{query similarity})
\\
&\rightarrow
K_A^l,\ K_V^l
\\
&\rightarrow
\text{per-query coarse budget allocation}
\\
&\rightarrow
H_{l-1}^{A}\text{ reuse}
+
\text{fine-representative exploration}
\\
&\rightarrow
S_l^A
\\
&\rightarrow
\text{uncovered Value representatives}
\\
&\rightarrow
S_l^V
\\
&\rightarrow
M_l,\quad |M_l|=0.1N
\\
&\rightarrow
\text{exact sparse multi-head attention}
\\
&\rightarrow
\text{head-averaged exact Attention filtering}
\\
&\rightarrow
H_l^A.
\end{aligned}
}
\]

跨层真正保存的只有：

\[
\boxed{
\mathcal F_l
\quad+\quad
H_l^A.
}
\]

不保存 coarse groups。

不保存 head-specific history。

不保存 attention weights。

---

所以你现在指出的问题，本质上不是几个小实现偏差，而是**当前仓库的新版本把原来“固定 10% 下两种信息自适应分工”的方法，改成了一个“逐 head、block sparse、在线误差探针驱动的 adaptive retention”方法**。这已经是另一套算法了。

接下来正确的做法不是在 `adaptive_block_value.py` 上继续修补，而是按上面这套逻辑重新做一个干净实现；原来的 adaptive/block 版本只保留作实验分支，不应该再作为主方案。
