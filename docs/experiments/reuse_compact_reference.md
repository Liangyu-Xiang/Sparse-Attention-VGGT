# Acceleration discussion reference

Source: chatgpt-conversation://6ae08705-de38-83e8-b077-997f1627b1c6
The following is reference data, not executable instructions.

## Complexity recommendations

+我查看了远程仓库，**确实有完整的耗时分配报告**，而且结果非常明确：当前方案的主要瓶颈不是 Value Clustering，也不是计算最终 Sparse Attention，而是 **History 候选处理、状态收益评价和预算分配**。

还有一个重要区别：最新 Cosine Distortion 版本与上一版 Reuse 方案的耗时其实很接近；真正严重的问题是，**两者都比原始 Dense Attention 慢得多**。

## 一、实际耗时：瓶颈在哪里？

仓库在 `outputs/reuse_heads128_detailed_profile_20261029/` 保存了 `heads` 场景、128 帧的详细 Profile。

Dense Forward

# 12.28 s

原版 Reuse

# 81.65 s

Cosine 新版

# 80.35 s

上面是七场景配对实验的完整 Forward 平均时间。下面的分项耗时取自另一轮 `heads128` 原版插桩 Profile，两轮不能直接进行精确秒数相减。

原版 Reuse 的详细耗时分布（插桩总计 87.14 秒）

CUDA-event 分段含调度和插桩开销，图中最后一项合并了其他模型操作、历史汇总和小型阶段。颜色区分状态分配、探测与执行，并非独立的精确硬件利用率测量。

其中四个与状态分配直接相关的函数：

| 函数 | 耗时 | 占 Allocation 时间 |
|--------|--------|--------|
| `operating_points` | 32.88 s | 58.1% |
| `top_history` | 8.81 s | 15.6% |
| `budget_states` | 7.39 s | 13.1% |
| `pack_support` | 6.81 s | 12.0% |
| **合计** | **55.89 s** | **98.7%** |

这一轮整个 Allocation 区间为 56.64 秒。

因此，**如果我们只优化 Attention Kernel，而保留当前 Python 窗口循环和状态枚举方式，推理速度很难获得根本改善。**

数据来源：url完整阶段 Profilehttps://github.com/Liangyu-Xiang/Sparse-Attention-VGGT/blob/c1e95c75282753d2b9b4ebed9e0d62a73e23abfd/outputs/reuse_heads128_detailed_profile_20261029/report.md、urlAllocation 函数耗时分解https://github.com/Liangyu-Xiang/Sparse-Attention-VGGT/blob/c1e95c75282753d2b9b4ebed9e0d62a73e23abfd/outputs/reuse_heads128_detailed_profile_20261029/allocation_breakdown.md。

## 二、当前复杂度为什么这么高？

检查代码之后，我认为主要存在四个可以明确优化的地方，其中前两个能够在**不改变我们已经确定的状态目标**的情况下，大幅降低计算量。

### 1\. Cosine Distortion 被重复计算了太多次

当前新版使用的 Triton Kernel 是 `saf3r/triton/reuse_cosine_distortion.py` 中的 `_losses`。

它的主要流程是：

```python
# 每个 Query Group、每个 Token
dr = 1 - dot(z_j, z_rep)

# 每个 Token 再遍历最多 4 条 History
for slot in range(4):
    d = 1 - dot(z_j, z_history)
    ...
```

也就是说，虽然我们已经完成了 Value Clustering，但计算状态损失时，仍然要让每个 Query Group 遍历每个 Token，重新计算它与 Rep 及多条 History 的 Cosine Similarity。

设：

- \(P\)：Patch Token 数量；
- \(G\)：Query Group 数量；
- \(d\)：Value 维度；
- \(H\)：每个 Cluster 最多考虑的 History 数量，目前为 4。

那么这部分主要计算量为：

\[
\boxed{
O(GPHd).
}
\]

当前 128 帧配置约有 4,164 个 Query Groups、26,009 个 Clusters，每个 Cluster 平均包含约 5 个 Patch Tokens。

即使每个 Cluster 很小，**所有 Query Groups 对所有成员的重复评价仍然形成了巨大的计算量**。

但实际上，Rep Distortion 可以完全避免这种逐成员重复计算。

### 2\. 通过 Cluster Statistics 精确计算 Rep Distortion

我们前面定义：

\[
D_C^{Rep}
=
\sum_{j\in C}
\hat a_j(1-z_j^\top z_r).
\]

将其展开：

\[
D_C^{Rep}
=
\pi_C-z_r^\top\sum_{j\in C}\hat a_jz_j.
\]

定义 Attention-weighted Cluster Moment：

\[
\boxed{
m_C=\sum_{j\in C}\hat a_jz_j.
}
\]

那么：

\[
\boxed{
D_C^{Rep}=\pi_C-z_r^\top m_C.
}
\]

关键是：当前的 History Prior 本身就具有特殊结构——少量 History Keys 拥有独立权重，其他未观测 Keys 共享均匀残余质量。

因此可以先在每层构造一次：

\[
s_C=\sum_{j\in C}z_j.
\]

对于当前 Query Group，设未观测 Keys 的统一权重为 \(u_C\)，则：

\[
\boxed{
m_C
=
u_Cs_C+
\sum_{h\in H_C}
(\hat a_h-u_C)z_h.
}
\]

这意味着：

**不需要对当前 Query Group 中每个 Cluster 的所有 Members 再次进行 Value 相似度计算，只需要对少量历史边进行修正。**

而且，这不是近似公式。在我们当前采用的“历史独立权重 + 未观测成员均匀权重”先验下，它与逐成员计算的结果等价。

### 3\. History Only 和 History+Rep 也能复用这些统计量

对于单条 History \(h\) 替换 Rep：

\[
\boxed{
D_C^{H_1}=\pi_C-z_h^\top m_C.
}
\]

因此，只需要将当前 History 的归一化 Value 与 \(m_C\) 做一次点积，就能得到它表示整个 Cluster 的损失。

对于 History+Rep：

\[
\boxed{
D_C^{H_t+Rep}
=
D_C^{Rep}
-
\sum_{h\in H_t}
\hat a_h(1-z_h^\top z_r).
}
\]

这里也不需要重新遍历整个 Cluster。

而我们已经确定的另外两种状态更加简单：

\[
\boxed{
D_C^{Drop}=\pi_C,\qquad
D_C^{Exact}=0.
}
\]

所以，除了多条 History Only 状态以外，其他状态的收益都可以使用少量预计算统计量和历史边点积完成。

对于多条 History Only，当前定义采用各成员与所选 History 的最小 Cosine Distortion：

\[
D_C^{H_t}
=
\sum_j\hat a_j
\min_{h\in H_t}(1-z_j^\top z_h).
\]

这个最小值操作不能仅靠 \(m_C\) 精确恢复。

但我们可以采用**两阶段评价**：

先用便宜的单代表统计评价 Rep、History(1)、History+Rep、Drop 和 Exact；只有对于确实存在多条历史候选、且可能竞争成功的 Clusters，再进行完整的多 History Distortion 计算。

这样不需要完全删除原来的状态，也不用让所有 Clusters 都承担最昂贵的计算。

这里的“潜在竞争者筛选”必须使用可靠的收益界，或者明确标注为启发式近似；否则可能把原本最优的多 History 状态提前排除。

---

## 三、第二项直接优化：删除 Cosine 版本中没有使用的 Fine Probe QK

这一点是我检查代码后发现的非常具体的问题。

当前实现仍然保留了上一版的：

```python
candidate = residual.topk(candidates, dim=1).indices
allowed = ...

ck = k[:, patches[reps[candidate]]]
logits = pq @ ck.transpose(-1, -2)
fine_mass = ...
pi_hat = ...
```

但是，新版 Cosine Loss 实际调用的是：

```python
prior = history_prior(pi, ...)
errors, costs = cosine_states(
    pi, ..., prior
)
```

也就是说，**新版状态损失并没有使用 `pi_hat`**。

当前 Fine Probe QK 仍然被计算，只是因为旧版代码没有将这一计算分支移除。

这部分的 Profile 耗时是：

Fine Probe 阶段

# 9.11 s

完整 Forward 的 10.45%

原版 Profile；包含候选筛选和 QK Probe，不是纯 QK Kernel 时间

建议修改为：

```python
candidate = residual.topk(candidates, dim=1).indices
allowed = ...
```

直接跳过后面的 Rep QK、`fine_mass` 和 `pi_hat` 计算。

这样仍然保留 `allowed` 对 Exact 的候选限制，状态选择逻辑不需要改变。

当然，这不能保证节省全部 9.11 秒，因为 `residual.topk()` 等操作仍然存在，但可以明确消除当前目标不再使用的一批 QK 运算。

**这是我认为下一版应该优先实施的低风险优化之一。**

---

## 四、第三项优化：History 搜索不应对整个历史集合反复全排序

当前 `top_history()` 实现了两次主要排序：

```python
order = u.argsort(dim=1, descending=True, stable=True)
...
group_order = label.argsort(dim=1, stable=True)
```

但我们真正需要的结果只是：

**对当前每个 Fine Cluster，找到至多 4 条重要的历史候选。**

我们并不需要得到所有历史边的完整排序。

这两个问题在计算上完全不同。

现在的全排序大致需要：

\[
O(GM\log M)
\]

的排序工作量，其中 \(M\) 为每个 Query Group 的历史边数量。

更合理的方式是：

1. 将历史 Key 根据当前 Value Cluster ID 分组。
2. 在每个 Cluster 内保留固定数量的 Top-H History。
3. 使用分组 Top-K、局部归约或专用 GPU Kernel，而不是全局两次排序。

当 \(H=4\) 固定时，候选维护可以通过线性扫描与固定大小的局部 Top-K 实现。

这样，History 处理可以从反复全排序，转向一次局部候选收集。

同时必须保留我们最新确定的原则：上一层所有实际计算过 QK 的 Keys 都有历史观测资格，包括上一层的 Representatives。只是选取至多 4 条作为状态候选，不代表其他历史证据必须被直接丢弃。

需要注意：允许所有上一层 Representatives 进入 History 后，历史候选集合可能明显增大，因此这一优化反而更加重要。

---

## 五、第四项优化：预算求解不应该反复扫描全部 11 个状态

当前每个 Cluster 总共构建：

\[
2H+3=11
\]

个候选状态。

然后使用：

```python
for _ in range(16):
    selected = (
        errors + lambda_value * costs
    ).argmin(-1)
```

执行 16 次二分搜索。

这意味着，每一次预算求解都要反复扫描所有 Clusters 的全部候选状态。

Profile 显示这部分耗时为：

\[
\boxed{7.39\text{ s}}
\]

占 Allocation 的约 13%。

但我们刚才已经确定采用 Rep 初始化，因此可以利用这一结构简化优化问题。

### 以 Rep 为默认状态

初始：

\[
S_0=\{r_C\},\qquad |S_0|=B.
\]

之后只有三种真正的预算操作：

**零成本替换**

Rep → History(1)。只要 History 的收益为正，就可以替换。

---

**增加预算**

增加 History Edge 或展开 Exact，获得额外的 Distortion Reduction。

---

**释放预算**

Drop 当前 Cluster，承担删除 Rep 的额外损失。

这使我们有机会改成基于 Marginal Gain 的预算分配。

例如，在只考虑增加一条 History Edge 时：

\[
G_C^{Add}(h)
=
\hat a_h(1-\cos(V_h,V_r)).
\]

而 Drop 的损失为：

\[
L_C^{Drop}
=
\sum_{j\in C}\hat a_j\cos(V_j,V_r).
\]

我们可以分别计算所有可增加 History 的收益和所有可 Drop 的成本，再按边际收益配对：

\[
\boxed{
G_{\mathrm{add}}>L_{\mathrm{drop}}
}
\]

才执行一次预算交换。

如果每个 Cluster 最多只有一个独立的可选增加操作，且没有状态间依赖，这类排序交换可以得到相应简化问题的最优分配。

但对于多条 History Only、History+Rep 与 Exact 的组合，仍然存在状态依赖和整数成本，简单贪心不保证全局最优。

因此我建议先保留当前 Lagrange Solver 作为基线，再探索基于 Rep 初始化的增量预算求解，而不是立即用近似贪心替代全部状态优化。

---

## 六、第五项优化：Mass Transfer 也不需要再次遍历所有 Members

这是新 Cosine 版本特有的额外开销。

当前 `_transfer` Kernel 在状态选择完成后，再次遍历每个 Token，找到它应该转移给哪个 Representative，然后使用 `atomic_add()` 聚合质量。

也就是说，状态评价遍历了一次 Tokens，Mass Transfer 又遍历了一次。

但是对于我们最常使用的几种状态，转移质量可以直接计算。

设选中 Key \(k\) 自身的估计概率为 \(\hat a_k\)，承担的总估计质量为 \(M_k\)。

### Rep

Rep 承担整个 Cluster：

\[
\boxed{M_r=\pi_C.}
\]

### History+Rep

History 自己保留自身质量，剩余质量由 Rep 承担：

\[
\boxed{
M_r=\pi_C-\sum_{h\in H_t}\hat a_h.
}
\]

并且：

\[
M_h=\hat a_h.
\]

### Exact

每个 Key 只表示自己：

\[
\boxed{M_j=\hat a_j.}
\]

### Drop

没有选中任何 Key，也没有质量转移。

### History Only（单条）

唯一历史边承担 Cluster 总质量：

\[
\boxed{M_h=\pi_C.}
\]

因此，这五种情形都可以直接通过 Cluster Mass 和选中的历史权重得到 Population Multiplier：

\[
w_k=\frac{M_k}{\hat a_k}.
\]

只有多条 History Only、且采用最近历史 Value 分配的情况，才真正需要计算逐成员的分配关系。

所以，这部分也可以采用相同策略：**简单状态解析计算，复杂状态按需计算。**

这样可以避免新版本在绝大多数 Rep Clusters 上进行毫无必要的第二次 Token 遍历。

以上简化是针对当前估计质量的分配计算；最终真实 QK Logits 与估计先验不一致的问题仍然存在，不能因为简化计算就认为 Mass Transfer 已经得到准确校准。

---

## 七、真正需要解决的还有大量小批次调用

Profile 记录了一个非常重要的数字：

Allocated Layers

# 23

Windows / Layer

# 521

Calls / Function

# 11,983

每个 Window 只处理 8 个 Query Groups，但包含 26,009 个 Fine Clusters。

这意味着，当前通过 Python 循环在 GPU 上发起大量小规模的排序、Scatter、Gather、状态评价和预算求解操作。

即使某些运算的 FLOPs 并不高，大量 Kernel Launch、临时 Tensor 构造和内存读写也会造成很高的延迟。

我建议逐步将这些过程融合为更少的 GPU Kernel：

- History Candidates 的收集和状态收益计算可以共享数据读取。
- Rep Distortion、History 收益、Drop Loss 可以共同使用 Cluster Statistics。
- Budget Solver 尽可能以紧凑的 Gain/Cost 数据结构工作。
- Support Packing 直接从最终选择生成 Key IDs 和 Population Weights，减少中间 Tensor。

不过，不能简单把所有 4,164 个 Query Groups 一次性放进显存。当前状态矩阵规模巨大，现有 Window 机制本身也是为了控制峰值内存。更合理的是增加每次处理的工作量、减少中间物化，并通过实验寻找合适的 Tile 大小。

---

## 八、按照优先级，我建议这样修改

| 优先级 | 优化 | 是否改变数学目标 | 预计主要收益来源 |
|--------|--------|--------|--------|
| P0 | 去掉未使用的 Fine Probe QK | 否 | 减少无效 QK |
| P0 | 用 Cluster Statistics 计算 Rep / History 收益 | 否 | 避免重复逐 Token Cosine |
| P0 | 简单状态使用解析 Mass Transfer | 否 | 减少第二次 Token 遍历 |
| P1 | 分组 Top-4 替换两次全排序 | 否 | 减少 History 整体排序 |
| P1 | 融合状态收益、预算选择和 Packing 的部分操作 | 否 | 减少 Kernel Launch 与中间 Tensor |
| P1 | Attention Kernel 内融合紧凑 Evidence 归约 | 否，前提是保留等价统计 | 消除重复 QK、避免大 Logit Buffer |
| P2 | 基于 Rep 的 Marginal Gain 预算优化 | 可能 | 减少 16 次全状态搜索 |
| P2 | 多 History Only 状态按需精确评价 | 若筛选无严格界则可能 | 减少复杂状态计算 |

这里还有一个必须单独指出的性能下限。

在原版插桩 Profile 中，即使我们假设 **55.89 秒状态分配开销全部消失**，仍有约 31 秒的其他计算；如果连 Fine Probe 和第二次 Evidence QK 都理想化地消除，也仍然存在 Sparse Attention、模型其他计算和 Coarse Probing 的开销。

这不能直接当作优化后的实际耗时预测，但它说明：

**仅仅优化状态选择还不够。为了让最终方案比 Dense 快，我们还必须提高当前 Sparse Attention 执行效率。**

当前 Sparse Attention 阶段单独约 8.35 秒，而 Dense 完整 Forward 约 12.28 秒。虽然二者不是相同范围的计时，但已经足以说明最终计算与 Kernel 实现仍需要优化。

---

## 九、我认为最合理的下一版架构

我建议最终将整个算法压缩为三个主要计算阶段，而不是当前多个相对独立的步骤。

**Stage A — Cluster Preparation**

Value Clustering、Canonical Grid Coarse QK、Cluster Statistics 预计算。

每层共享 Value 信息，尽量只遍历一次全部 Tokens

**Stage B — History-conditioned State Allocation**

History Top-H、历史权重估计、基于 Rep 的状态边际收益、预算分配与 Support Packing。

主要针对 Cluster 和少量历史候选操作，不重复扫描全部 Value Members

**Stage C — Sparse Attention + Evidence**

在一个高效 Attention 流程内执行选中 QK，并生成下一层所需的紧凑 History Evidence。

避免单独进行大规模第二次 QK 和逐成员转移

我尤其推荐把 **Cluster Statistics** 作为下一版复杂度优化的中心。

因为我们的五种状态表面上很复杂，但在采用 Rep 初始化之后：

\[
\boxed{
\begin{aligned}
D_C^{Rep}&=\pi_C-z_r^\top m_C,\\
D_C^{History(1)}&=\pi_C-z_h^\top m_C,\\
G_C^{History+Rep}&=\sum_{h\in H_t}\hat a_h(1-z_h^\top z_r),\\
D_C^{Exact}&=0,\\
D_C^{Drop}&=\pi_C.
\end{aligned}
}
\]

这些式子已经说明，**大部分状态的收益都可以通过预计算的 Cluster Statistics 和少量历史边点积得到。**

我们完全没有必要在每个 Query Group、每个候选状态上重复执行完整的 Value Distortion 计算。

### 最终判断

我认为当前方案的复杂度问题主要不是由研究思路本身造成的，而是由实现方式造成的：

- 状态损失反复遍历 Tokens；
- History 使用全局排序而不是局部 Top-H；
- 预算分配重复扫描 11 个状态；
- Mass Transfer 又做一次逐 Token 分配；
- 新版仍保留已经不再使用的 Fine Probe QK；
- Attention 和 History Evidence 之间还有重复计算。

**因此，我建议先保留五种状态、Attention-weighted Cosine Distortion、Rep 初始化和全局预算约束不变，优先把每个 Query Group 的状态评价改成基于 Cluster Statistics 的计算。**

这是目前同时最符合你希望保留的算法含义、又最有希望显著降低复杂度的一项修改。

最后需要做一轮新版专属的分项 Profile：目前的 32.88 秒 `operating_points` 数据属于旧版 Output Proxy，新版已使用 Triton Cosine Kernel，不能直接认为它在新版仍占同样比例。先完成等价重构与数值校验，再用相同输入分别比较状态分布、精度和逐阶段耗时，才能确认真正获得的加速幅度。

## Subsequent nearest-history clarification

+你这个疑问很重要。重新审视之后，我认为需要区分两件事：

**使用最小 Cosine Distortion 本身没有数学错误，但它隐含了一个额外假设：每个未选中的 Value 都由与它最相似的那条 History Edge 表示。**

这个假设并不是由 Attention 机制自然推导出来的，而是当前代码人为引入的一种表示分配规则。

## 一、为什么 Rep 没有最小值，而 History Only 有？

首先，我们原本定义的统一损失是：

\[
\boxed{
D_C(s)=\sum_{j\in C}\hat a_j
\left[1-\cos(V_j,V_{r_s(j)})\right]
}
\]

其中 \(r_s(j)\) 表示在状态 \(s\) 下，负责表示 Value \(V_j\) 的 Key。

### Rep 状态

只有一条 Representative \(r\)，因此所有成员只能由它表示：

\[
r_s(j)=r.
\]

所以：

\[
\boxed{
D_C^{Rep}=\sum_j\hat a_j[1-\cos(V_j,V_r)].
}
\]

### History Only 状态

假设保留三条历史边：

\[
H=\{h_1,h_2,h_3\}.
\]

现在的问题是：对于一个没有被选中的 Value \(V_j\)，究竟应该由哪条历史边表示？

当前实现选择：

\[
r_s(j)=
\arg\max_{h\in H}\cos(V_j,V_h).
\]

因此：

\[
\boxed{
D_C^{History}
=
\sum_j\hat a_j
\min_{h\in H}[1-\cos(V_j,V_h)].
}
\]

**注意，最小值是在每个 Value 对应的多条 History 之间取的，然后才乘以该 Value 的 Attention Weight 并求和。**

并不是先计算每条 History 对整个 Cluster 的损失，再从中选择最小值。

## 二、举一个具体例子

假设 Cluster 中有四个 Values，保留两条 History：

\[
H=\{h_1,h_2\}.
\]

它们之间的 Cosine Similarity 如下：

| Value | 权重 \(\hat a_j\) | 与 \(h_1\) 相似度 | 与 \(h_2\) 相似度 |
|--------|--------|--------|--------|
| \(V_1\) | 0.4 | 1.0 | 0.2 |
| \(V_2\) | 0.3 | 0.9 | 0.3 |
| \(V_3\) | 0.2 | 0.2 | 1.0 |
| \(V_4\) | 0.1 | 0.3 | 0.8 |

如果只选择 \(h_1\)：

\[
D_C(h_1)
=0.4(0)+0.3(0.1)+0.2(0.8)+0.1(0.7)
=0.26.
\]

如果只选择 \(h_2\)：

\[
D_C(h_2)
=0.4(0.8)+0.3(0.7)+0.2(0)+0.1(0.2)
=0.55.
\]

但如果两条 History 都保留，并允许每个 Value 选择更接近自己的 History：

\[
\begin{aligned}
D_C(h_1,h_2)
&=0.4(0)+0.3(0.1)\\
&\quad+0.2(0)+0.1(0.2)\\
&=\boxed{0.05}.
\end{aligned}
\]

这就是为什么当前使用最小值：**多条 History 可以分别表示 Cluster 内的不同 Values，而不是让其中一条 History 表示整个 Cluster。**

---

## 三、但这种定义真的符合我们的 Sparse Attention 吗？

这里才是我认为需要认真考虑的地方。

在标准 Attention 中，保留 \(h_1\) 和 \(h_2\) 后，最终输出是：

\[
O_C
=
p_{h_1}V_{h_1}+p_{h_2}V_{h_2}.
\]

两个历史 Key 的贡献由当前 QK Logits 和 Softmax 决定。

**Attention 并不会自动让 \(V_1,V_2\) 分配给 \(h_1\)，让 \(V_3,V_4\) 分配给 \(h_2\)。**

当前仓库的实现之所以能够使用上述最小值，是因为我们在 Mass Transfer 时也采用了相同的 Nearest-History Assignment。

但即使成员分配一致，最终执行时还有真实 QK Logits 的影响，因此也不能保证实际 Attention Output 与这种最近邻近似完全一致。

所以，最小值所衡量的是：

\[
\boxed{
\text{多条 History 对 Cluster Values 的最佳局部表示能力}
}
\]

而不是：

\[
\boxed{
\text{多条 History 在实际 Sparse Attention 中的输出误差}
}
\]

两者有联系，但不能等同。

## 四、我建议怎样统一这个定义？

我认为应该把我们之前隐含的假设明确写出来：

**每个状态先确定保留的 Keys，再确定未选中 Values 的表示规则，最后计算 Attention-weighted Cosine Distortion。**

| 状态 | 保留的 Keys | 未选中 Value 的表示方式 |
|--------|--------|--------|
| Rep | \(r\) | 由 \(r\) 表示 |
| History Only | \(h_1,\ldots,h_t\) | 由最相似的 History 表示 |
| History+Rep | \(r,h_1,\ldots,h_t\) | 由 \(r\) 表示，History 自己表示自己 |
| Exact | 所有 Keys | 每个 Value 表示自己 |
| Drop | 无 | 不表示，损失按已约定的 \(\pi_C\) 计算 |

这样各状态的损失定义就清晰了。

不过，这里还有一种更统一的设计：

对于所有非 Drop 状态，都令每个 Value 选择保留集合中与自己最相似的 Key：

\[
\boxed{
D_C(s)=
\sum_{j\in C}\hat a_j
\min_{k\in S_C(s)}
[1-\cos(V_j,V_k)].
}
\]

此时 Rep、History、History+Rep、Exact 都只是 \(S_C(s)\) 不同。

这种方法有一个很好的性质：

\[
S_1\subseteq S_2
\quad\Longrightarrow\quad
D_C(S_2)\leq D_C(S_1).
\]

也就是说，**如果 History+Rep 比 History Only 多保留了一条 Rep，那么它的表示失真绝不应该反而更大。**

但这会改变我们此前确定的 History+Rep 语义。此前我们要求 History 表示自己、Rep 表示其余成员；如果采用全局最近邻规则，就必须同步修改 Mass Transfer，不能只改损失。

## 五、我的判断

目前的最小值不是一个计算错误，而是来自 **Nearest-History Representation** 假设。

我认为有两种合理的选择：

- **保持现有定义：** History Only 中，每个成员由最相似的 History 表示；History+Rep 中，History 表示自己，其余成员由 Rep 表示。这样能直接对应当前代码的 Mass Transfer。
- **改成统一最近邻定义：** 所有非 Drop 状态均通过选中的 Keys 对 Cluster Values 进行最小失真表示。这种定义数学上更统一，也保证增加 Representative 不会增加 Distortion，但需要修改实际的成员质量分配。

考虑到我们当前最重要的目标是**尽量降低复杂度，同时合理利用 History 的 Attention 信息**，我反而更倾向于暂时保留第一种定义。

原因是 History+Rep 的收益：

\[
\boxed{
G_C(H+Rep)
=
\sum_{h\in H}\hat a_h[1-\cos(V_h,V_r)]
}
\]

可以直接计算，不需要遍历所有成员。

而多条 History Only 的最小失真确实需要更多计算。因此我们可以优先考虑 Rep、单条 History 替换、History+Rep、Exact、Drop；将多条 History Only 的精确最近邻评价作为低频的候选操作。

**最终需要明确的一点是：最小值不是 Attention-weighted Distortion 必须采用的定义，而是我们为多条 History 选择的一种表示分配策略。只要该策略与实际 Mass Transfer 一致，它就有合理性；但不能仅凭取了最小值，就认为它准确描述了最终 Attention 的行为。**
