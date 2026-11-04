# History 状态：历史 Key 集合并入 Rep

采用用户截图中的残差表示定义，并对所有簇显式使用以下执行支持集：

\[
S_H(C)=H_C\cup\{r_C\}.
\]

其中 H_C 是当前簇内的历史候选 Key 集合，r_C 是当前 Rep。并集去重，r_C 无论是否曾在历史中出现，都只执行一次。该定义对应现有 A 路径（`residual_rep`），本轮为它增加明确命名的配置，并核验执行边界；配置逐字节等同于已经评测过的 A 配置。

配置：`configs/evaluation/reuse_shared_value_history_union_rep_256.json`。

## 损失

\[
L_R(C)=\sum_{j\in C}\hat p_j\|V_j-V_r\|_2^2,
\]

\[
L_H(C)=L_R(C)-\sum_{h\in H_C}\hat p_h\|V_h-V_r\|_2^2
      =\sum_{j\in C\setminus(H_C\cup\{r\})}\hat p_j\|V_j-V_r\|_2^2.
\]

不再要求 r∈H 才能使用这个定义。若 r∉H，将 r 并入最终支持即可；因为其自身失真项为0，这不会给损失新增一项。普通历史成员表示自身，剩余成员由 Rep 表示。

## 成本与执行权重

预算成本必须与实际发出的唯一 Key 数量一致：

\[
\operatorname{cost}_H=|H_C\cup\{r\}|=|H_C|+\mathbf1[r\notin H_C].
\]

该成本在状态选择前进入预算求解，不在分配完成后额外追加 Rep。继续保持含 Special 的20%总 Key上限。

令 O=H_C\{r}，普通历史成员执行权重为1，Rep执行权重为：

\[
w_r=\frac{\pi_C-\sum_{h\in O}\hat p_h}{\hat p_r}.
\]

在簇内均匀概率假设下，w_r=|C|−|O|。因此历史成员只承担自身的概率，剩余质量明确由 Rep 承担。若 Rep 本身也是历史 Key，它在这里以 Rep 的角色执行一次。

## 边界

| 历史集合情况 | 最终支持 | 成本 | 语义 |
|---|---|---:|---|
| r∈H | H | len(H) | 保留已有Rep，去重 |
| r∉H，H非空 | H∪{r} | len(H)+1 | 增加Rep，并预先计入预算 |
| H={r} | {r} | 1 | 与Rep状态相同的损失及权重 |
| H为空 | Rep候选仍可用 | 1 | 不额外建立重复History候选 |
| H∪{r}=C | C | len(C) | 权重全部为1，等价完整保留 |

普通History的集合为 O；即便 H未包含Rep，选出的History状态仍一定包含当前Rep。对于每个已选择History状态的簇，执行器保证Rep出现一次。

## 实现与验证

执行复用 `saf3r/patch/state_loss_controller.py` 的 `residual_rep` 分支，以及 `saf3r/triton/state_loss_controller.py` 的损失/权重内核。概率、Drop/Rep/Exact、中心化Drop和8维输出特征沿用现有A配置；本轮只明确选择History定义，不以用户截图为由悄悄调整这些设置。

新增CUDA测试 `test_history_support_is_union_with_rep_and_costs_unique_keys` 检查Rep已经在历史、不在历史、History只有Rep、普通History权重为1、真实唯一Key成本、严格含5个Special的20%预算，以及图示减法损失与执行支持一致。测试通过，未增加生产QK或新执行内核。

这是现有A方法的命名及边界验证，不是新一轮精度修复结果。复用A已有的7Scenes七场景和ETH3D三场景完整评测；不因配置别名重跑未变更方法。新增测试只属数值诊断，六项质量均N/A。详见 `outputs/history_union_rep_20261104/report.md`。
