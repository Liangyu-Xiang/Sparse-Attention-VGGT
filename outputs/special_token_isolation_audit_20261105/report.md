# 特殊 token 双向保护审计

旧布局修复版与已淘汰的 partial-states 版均配置 dense_special_queries=1。
每帧为1个camera token与4个register token。它们已从patch聚类/稀疏选择域中排除，始终作为完整Key参与所有Query的注意力，并且不应用patch的population补偿。然而，旧版本只有camera Query对完整Key集合计算；register Query仍对稀疏Key集合计算。

新增两个独立配置，将 dense_special_queries 从1改为5，其余参数完全保留：

- configs/evaluation/reuse_layout_fixed_all_specials_dense.json
- configs/evaluation/reuse_partial_states_all_specials_dense.json

目标输出支持关系：

| Query / Key | 特殊Key | 普通patch Key |
|---|---|---|
| camera/register Query | 全部 | 全部 |
| 普通patch Query | 全部 | 按原方法稀疏选择 |

复用已有 correct_output 全量SDPA分支，覆盖全部camera/register Query的最终输出。这保证输出支持关系，但不是将这些Query从底层稀疏kernel中物理剔除：稀疏observer仍先执行并作为历史流，随后全量分支覆盖特殊Query输出。因此存在重复计算，不能据此承诺完整前向不变或加速。所有已有精度/速度成绩均属于旧配置，不转记为新配置成绩。

GPU4合成双帧算子诊断通过：特殊Key与patch域不交；所有普通Query都能接收特殊Key的Value；一个被普通稀疏支持排除的patch Key仍能影响全部camera/register Query；特殊Query输出与完整注意力参考匹配（最大绝对误差约1.75e-5，FP16）。这仅验证算子语义，不是完整模型或数据集评测；AUC@3、AUC@30、ACC、Comp、CD、F1及相对Dense完整前向加速均为N/A，原因见quality.json/quality.md。

特殊Query全量保护是否能改善ETH3D性能，仍需同输入的完整模型对照评测确认。当前证据不支持将其称为已经确认的性能退化根因。
