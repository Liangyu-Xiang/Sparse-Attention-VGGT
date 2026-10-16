# Online layer-shared Value representatives for VGGT global attention

实现入口：`saf3r/patch/value_evolving_attention.py`；标准模型配置：`configs/evaluation/eval_layer_value_local_vggt.yaml`，patch类型`layer_value_local`。

## 在线路径

- 自然编号global第1、2层维持dense，第10–24层也维持dense；全部frame attention不变。
- 第2层对本次前向的16头原始V按头取FP32平均，独立全局PAM，K=round(0.1×全部帧patch数)。选两个固定种子全局eager PAM中目标较低且单交换收敛的解。
- 第3层用第2层的划分、medoid token位置和平均V构建固定q=8近邻图，做两轮同步reassignment和精确簇内medoid更新。
- 第4层继承第3层本次在线更新的结果，依此递归至第9层。不是对每一层重新引用dense缓存，也不做跨头匹配。
- 每个token在本层两轮更新中始终使用原始上一层标签的邻域；上一层medoid token锁定为该簇成员锚点，保持K不变。到下一层才以新划分和新medoid构造新的邻域。
- 各头共享一套medoid token索引。保留全部Query，Key与Value同步gather同一批代表patch；并保留全部camera/register special keys。special queries也只访问同一批保留keys，没有额外dense special-query分支。
- attention softmax直接在选定K/V上计算，不增加簇大小权重，不将不同头的Value向量用于实际输出平均。平均V仅用于决定共享代表位置。
- 每次输入由aggregator pre-hook重置状态，防止序列间泄漏。仅支持batch=1的推理，不支持训练。

`retention=1`时选回全部patch，单元测试核对恢复整个24层toy模型的dense输出；低保留率测试核对与显式注意力掩码等价、移除的V变化不影响输出。
标准模型配置可覆盖`neighbors`、`iterations`、`sparse_start/end`、`seed_backend`；默认q8、2轮、L3–L9。

## 初始化内存

`seed_backend=device`在当前设备构建分块完整距离矩阵；`host`将完整FP32距离矩阵放主存，GPU只计算分块及交换，保留同一全量候选搜索定义。
默认host适合100帧：N=103600，单距离矩阵约42.93 GB，避免它与模型及中间激活争用显存。不抽样token或改变输入帧数。
聚类逻辑移入可安装的`saf3r.patch.value_clustering`和`value_medoids`；原分析脚本保留兼容导入，同一套实现可继续用于既有离线实验。

## 已完成10帧验证

Chess/seq-01，第0、10、…、90帧，原始权重`model_tracker_fixed_e20.pt`，BF16前向，10%保留，q8，两轮。
第2层全局seed的平均distortion=0.169262；L3–L9各层使用1036个代表patch keys + 50个special keys，全部10410个queries均保留。
相邻层来源核对为2→3→4→5→6→7→8→9，固定锚点和簇数均保持。
额外比对同输入的dense Value：第2、3层平均V相对L2差约1e-9；第4层为9.12%、第9层为11.60%。这确认后续层使用的是稀疏反馈后的实际特征，而非离线dense Value缓存。

| 指标 | Dense | 在线递归局部方案 |
|---|---:|---:|
| Pose AUC30 | 0.773333 | 0.772593 |
| Pose AUC15 | 0.579259 | 0.577778 |
| Pose AUC05 | 0.080000 | 0.084444 |
| Pose AUC03 | 0.029630 | 0.037037 |
| 单次端到端耗时（秒） | 1.814 | 7.569 |

相对dense原始深度预测的平均相对变化为0.4825%；这是未经尺度对齐的预测差异，不是对GT的深度误差。
耗时包括本次在线PAM、CPU局部更新和诊断保存；第2层PAM约5.11秒，7次局部选择共约1.53秒。当前端到端实现尚未加速，不将注意力keys减少解释为已经获得实际推理加速。
这是单一输入的功能/数值验证，不是完整基准或重复计时统计。

结果：`outputs/layer_value_local/chess_n10_s10_q8/result.json`。
各层代表、簇归属、锚点、两轮快照及实际在线平均V：`selection/forward_0001/layer_02`至`layer_09`。

## 运行

使用已安装torch的环境（本次为`/data/mmc_syang/miniconda3/envs/fastvggt/bin/python`）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=4 python scripts/run_layer_value_local.py \
  --frames 10 --stride 10 --seed-backend device

# 五序列100帧，GPU1/2/3排队运行，每个序列分别执行dense与在线方案
PYTHONPATH=. python scripts/batch_layer_value_local.py --gpus 1 2 3
```

100帧状态：`outputs/layer_value_local/n100_s10_q8/status.json`；全部完成后生成同目录`summary.csv`。
标准评估配置沿用评估器的uniform帧采样；需要严格第0、10、…帧的复现时使用上述脚本的`--stride 10`。
该在线实验独立于此前1%离线Recall及10%单步distortion任务，保留它们已有的运行与输出。

## ETH3D

同一在线方案的dense/recursive_local配对测试：

```bash
PYTHONPATH=. python scripts/batch_layer_value_local.py --dataset eth3d --frames 100 --gpus 1 2 3
```

使用仓库ETH3D读取器的11个场景、标定图像顺序和已知问题视图过滤规则；直接使用world-to-camera GT，不能再求逆。
ETH3D不是视频，不套用7Scenes的stride=10；按标定顺序均匀选择最多100张，当前各场景14–76张，因此实际使用全部有效图像。
对应张数：courtyard 38、electro 39、kicker 31、pipes 14、relief 19、delivery_area 40、facade 76、office 26、playground 32、relief_2 20、terrains 42。
每个场景的两种模型使用完全相同的图像和GT；其余聚类、注意力、权重与BF16设置不变。
统计位姿AUC30/15/5/3、对dense深度预测的相对变化、包含在线聚类与诊断保存的推理耗时，以及峰值已分配显存。
这里不将预测差异当作GT深度误差，也不将位姿测试当作ETH3D官方3D重建评测。
状态及逐场景结果保存在`outputs/layer_value_local/eth3d_max100_uniform_q8/`，全部完成后生成`summary.csv`。

### 同输入SAF3R L3--L9对比

```bash
PYTHONPATH=. python scripts/batch_layer_value_local.py --dataset eth3d --scheme saf3r \
  --topk 128 --frames 100 --gpus 1 2 3 4 5 6 \
  --output-dir outputs/layer_value_local/eth3d_saf3r_l3_9_topk128
```

仅替换自然编号global第3--9层（代码索引2--8），各头沿用原始ETH3D校准profile；其余global层、frame attention和aggregator前向不变。
每个场景重新运行dense作为配对参考，输入与前述在线Value实验一致；使用全部11场景377张有效图像。
重要：原始profile在这些层没有`dino_topk`头，因此`topk=128`只是配置值，不会参与实际计算。
query-probe头预算仍由profile的`stride=128`控制，其余stride/首帧/帧内模式均不改变。
这不是“全部头每个query保留128个key”的实验，也不能按该预算解释结果。
输出metadata记录完整受限profile与`topk_effective=false`；单次计时包含选择开销、不含数据加载和模型加载。
