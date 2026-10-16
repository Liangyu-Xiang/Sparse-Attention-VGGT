# 10帧/10%：后层选择最佳前层头，以及先平均V后的层间ARI

继续使用Chess/seq-01第0、5、…、45帧、前三个VGGT global layers、每层16头。
每份划分都在全部10360个patch上做10% k-medoids，K=1036，special tokens排除。

## 1. 正确的匹配方向

令A[g,h]为前层头g与后层头h的完整Value簇划分ARI。
对每一个后层头h独立选择：

`best_previous(h) = argmax_g A[g,h]`

即在矩阵每一列上取最大值，不按相同编号匹配；允许多个后层头选中同一前层头，
不施加一对一约束。这里的“最优”仅指当前输入上ARI最大，不是功能性对应或可泛化结论。
ARI本身对称，但“每列取最大再平均”与之前“每行取最大再平均”并不相同。

此前完整16×16矩阵没有限制为同编号头对，但图中原先标注的是每行最优，方向与
用户要求相反。现已修正为每列最优，加粗并用红框标记；CSV给出全部32个选择。

- 第2层H02、H04选择第1层H03；其余14个头选择第1层H10。
- 第3层H03选择第2层H12，H12选择第2层H07；其余14个头选择第2层H02。

## 2. 先平均16头V，再独立聚类

每层对同一patch位置的16个原始64维V逐坐标取算术平均，FP32累加，得到
`[10360,64]`的平均V，再独立做K=1036的PAM，最后比较平均V簇划分的ARI。

`mean_V[l,i,d] = sum_h V[l,h,i,d] / 16`

不是平均原有簇编号，不是平均256个头对的ARI，也不是平均attention概率或输出。
不做L2归一化、头特征基底对齐或模型训练。头独立学习特征基底，逐坐标平均
具有基底依赖性，可能抵消特征，不能直接解释为多头共识划分。

GPU1执行两次初始化的PAM，最大3000次最佳单交换，三层全部收敛，选中解
分别交换1079、1052、1060次；保证容差下单交换局部最优，不保证全局最优。

## 3. 并列结果

| 相邻层 | 每后层头选最佳前层头，再平均ARI | 先平均16头V，再聚类的层间ARI | 全256头对ARI均值（参考） |
|---|---:|---:|---:|
| 第1→2层 | 0.240558 | 0.301342 | 0.212133 |
| 第2→3层 | 0.433264 | 0.396125 | 0.313457 |

前两项是不同构造上的一致性统计，不是同一划分的两种评价，也不能直接用于
断言哪种稀疏注意力的输出或下游效果更好。第三列参考值与“先平均V”的结果严格区分。
结论仅限此次输入。

## 4. 输出与验证

在`outputs/value_medoids/chess_n10_s5/`中：

- `all_head_ari_10pct/best_previous_heads.csv`：32个后层头的最佳前层头、ARI及同编号参考值。
- `all_head_ari_10pct/index.html`、`L00_L01_ari.png`、`L01_L02_ari.png`：修正匹配方向的矩阵。
- `head_mean_10pct_ari/index.html`、`adjacent_layers.csv`、`summary.json`：两种方法并列表、明确的计算定义。
- `head_mean_10pct_ari/mean_values.npz`：三层平均后的原始V。
- `head_mean_10pct_ari/layer_XX/clusters.npz`及`clustering.json`：代表点、完整归属、距离、两次初始化收敛细节。

验证包括32个选择全部等于对应列的argmax；三层平均V精确等于16头FP32均值；
三组平均V聚类全部归属用FP64距离核验；两项层间ARI独立对照sklearn。
15项相关单元测试通过，包含矩形矩阵下的方向测试以及允许前层头复用的测试。

```bash
PYTHONPATH=. python scripts/report_value_head_ari.py \
  --input-dir outputs/value_medoids/chess_n10_s5 --ratio 0.10
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/analyze_head_mean_value_ari.py \
  --input-dir outputs/value_medoids/chess_n10_s5 --ratio 0.10 --max-swaps 3000
```
