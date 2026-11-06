# 独立RGB相机加权代理筛选

此轮不是目标数据集质量评测，六项AUC3/AUC30/ACC/Comp/CD/F1均N/A。只使用已有ScanNet64和TUM64 RGB，当前稀疏轨迹同QKV的局部Dense Attention作为数值参照。

每层取4个确定性相机Query的完整Key概率，Head/Query平均，给前一层留出Patch Query的Wo输出误差加权；再按该层Patch总质量加权。它是启发式敏感性代理，不是真实相机损失或精确跨层Jacobian。末层Patch输出不直接进入CameraHead，故相机代理覆盖L0–22，普通Attention代理仍覆盖24层。

预先要求两序列相机代理都改善、两序列普通Attention误差均不比首轮参考恶化超过2%，在符合者中按相机代理相对比排序。共30个层间预算/簇数组合，全部保持整体10%总边上限。

**结果：没有合格候选，因此没有追加目标质量评测，也没有按测试分数挑选其他预算。该局部代理不足以保证真实相机AUC。**

| 候选 | ScanNet相机代理 | TUM相机代理 | 相机代理相对均值 | 两序列均改善 | 普通Attention保护 | 合格 |
| --- | --- | --- | --- | --- | --- | --- |
| middle12_c08 | 0.01762764 | 0.01572440 | 1.000000 | False | True | False |
| scaled_middle10_c06 | 0.01865939 | 0.01582948 | 1.032607 | False | False | False |
| middle12_c06 | 0.01930451 | 0.01666355 | 1.077427 | False | False | False |
| uniform10_c08 | 0.01889495 | 0.01719305 | 1.082646 | False | False | False |
| back12_c08 | 0.02126981 | 0.01569523 | 1.102381 | False | False | False |
| front12_c08 | 0.02034311 | 0.01715789 | 1.122605 | False | False | False |
| scaled_middle10_c08 | 0.01715819 | 0.02023388 | 1.130075 | False | False | False |
| middle14_c06 | 0.01911671 | 0.01875935 | 1.138741 | False | False | False |
| uniform10 | 0.02010060 | 0.01853754 | 1.159596 | False | False | False |
| front12_c06 | 0.02139836 | 0.01794664 | 1.177617 | False | False | False |
| uniform10_c06 | 0.02127895 | 0.01810380 | 1.179227 | False | False | False |
| back12_c06 | 0.02281737 | 0.01821698 | 1.226463 | False | False | False |
| front14_c06 | 0.02039307 | 0.02098223 | 1.245627 | False | False | False |
| back14_c06 | 0.02488128 | 0.01797664 | 1.277362 | False | False | False |
| back12_c10 | 0.02129977 | 0.02237651 | 1.315680 | False | False | False |
| back14_c08 | 0.02258871 | 0.02193854 | 1.338314 | False | False | False |
| middle12 | 0.02088151 | 0.02415383 | 1.360331 | False | False | False |
| back16_c08 | 0.02633599 | 0.01969619 | 1.373302 | False | False | False |
| front16_c06 | 0.02549564 | 0.02198729 | 1.422318 | False | False | False |
| front12_c10 | 0.02250293 | 0.02467530 | 1.422903 | False | False | False |
| front16_c08 | 0.02622782 | 0.02252389 | 1.460148 | False | False | False |
| back16_c06 | 0.02731471 | 0.02214223 | 1.478842 | False | False | False |
| back16_c10 | 0.02522804 | 0.02620938 | 1.548980 | False | False | False |
| front14_c08 | 0.02263664 | 0.02913581 | 1.568530 | False | False | False |
| back14_c10 | 0.02532874 | 0.02788972 | 1.605267 | False | False | False |
| front16_c10 | 0.02482498 | 0.02900881 | 1.626563 | False | False | False |
| middle14_c08 | 0.01827618 | 0.03520230 | 1.637748 | False | False | False |
| scaled_middle10 | 0.02224184 | 0.03407495 | 1.714385 | False | False | False |
| front14_c10 | 0.02763399 | 0.05690709 | 2.593341 | False | False | False |
| middle14 | 0.02561369 | 0.06699556 | 2.856826 | False | False | False |
