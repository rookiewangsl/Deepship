# 拖曳式 DAS 分布式光缆水听器：概念与 Deepship 接入摘要

## 一句话结论

拖曳式 DAS（Distributed Acoustic Sensing）光缆可形成高密度、长基线的分布式水听阵列，但其直接测量量是**光纤轴向应变或应变率**，不是传统水听器直接测得的**声压**。因此，缆体结构、标距、入射角、拖曳姿态、耦合和流噪共同决定数据分布；将传统水听器音频直接迁移到 DAS 时会产生显著域偏移。

## 1. 声—光—电转换物理链路

```text
水中声压 p
  -> 缆体/弹性增敏结构形变
  -> 光纤轴向应变 ε_parallel
  -> 有效折射率与光程改变
  -> 瑞利后向散射相位变化
  -> 相干光电探测与 I/Q 解调
  -> 差分相位、应变或应变率时空数据
```

相干激光脉冲沿光纤传播。光纤内部无数微小折射率扰动产生瑞利后向散射；外界应变改变各散射单元的相对相位。通过回波往返时间定位扰动位置：

\[
z = \frac{ct}{2n_g}
\]

对于标距为 \(L_g\) 的差分相位测量，常用近似为：

\[
\Delta\phi \approx \frac{4\pi n_\mathrm{eff}}{\lambda}(1-p_e)L_g\bar{\epsilon}_\parallel
\]

其中 \(n_\mathrm{eff}\) 为有效折射率、\(\lambda\) 为激光波长、\(p_e\) 为有效光弹系数。实际系统输出的量可能是相位、应变或应变率，具体标定系数取决于 interrogator 的解调定义。

关键点：声压到应变的传递函数 \(H_{p\rightarrow\epsilon}(f,\theta,z)\) 不是常数，它随频率、来波角、缆体结构、张力、增敏材料和环境耦合改变。因此 DAS 不应在未经校准时等同于全向声压水听器。

## 2. 技术分类

这些分类维度相互独立，不能混为一谈。

| 分类维度 | 类型 | 说明 |
|---|---|---|
| 光纤水听器原理 | 干涉型（Michelson/Mach–Zehnder）、FBG、Fabry–Pérot、DAS | 前三者通常是点式或准分布式；DAS 利用沿纤瑞利散射实现连续分布式测量。 |
| DAS 解调方式 | \(\phi\)-OTDR、coherent OTDR、OFDR | 拖曳和长距离应用通常采用相敏/相干 OTDR；OFDR 空间分辨率高但量程较短。 |
| 拖缆机械结构 | 直纤/普通缓冲缆、连续螺旋增敏缆、离散弹性单元缆、增强瑞利散射纤 | 机械结构直接塑造压力到应变的传递函数、方向性和流噪特性。 |

当前拖曳 DAS 水听缆的主要结构路线：

1. **直纤或普通缓冲缆**：光纤沿缆轴布置，主要对轴向应变敏感；适合有良好轴向耦合的场景，但横向入射与纯压力响应往往较弱。
2. **连续螺旋缠绕增敏缆（HW-TSC）**：光纤连续绕在柔性芯轴上，把横向形变转化为更多光纤长度变化，从而提高连续分布式声压响应。
3. **离散弹性单元缆（DS-TSC）**：光纤在多个弹性圆柱上绕制为增敏单元，单元之间保持低敏。光学读取仍是 DAS，机械上更像传统离散水听器阵列。
4. **增强散射光纤**：通过飞秒激光写入等方式增强瑞利散射，以提高回波强度、降低相位噪声；通常可与上述机械增敏方式组合。

## 3. 与传统压电水听器的区别

| 项目 | 传统压电水听器 | 拖曳 DAS 光缆 |
|---|---|---|
| 直接测量量 | 声压 \(p\) | 轴向应变/应变率 |
| 换能机制 | 压电材料受压产生电荷或电压 | 应变调制光相位，再经光电探测转换为电信号 |
| 方向性 | 常接近全向，且通常已校准 | 有强轴向方向性；直缆对垂直入射可出现响应弱或零陷 |
| 阵列形态 | 离散物理阵元 | 连续虚拟通道；相邻通道通常相关而非独立 |
| 幅值标定 | 声压灵敏度体系成熟 | 压力—应变响应依赖缆、部署与频率，需现场标定 |
| 水下硬件 | 每阵元常含传感与复用部件 | 感知缆内可减少阵元电子及复杂光器件 |
| 主要系统成本 | 阵元数和布放成本 | interrogator、数据传输/存储与实时处理成本 |

## 4. 优点与限制

### 优点

- 一根光纤即可形成高密度长阵列，有利于到达时差估计、波束形成、目标定位和轨迹跟踪。
- 水下感知段可减少有源电子与复杂光学元件，抗电磁干扰，潜在制造与维护更简化。
- 同时可观测船噪、缆振动、地震、海浪和流场等多种扰动。
- 空间连续性优于稀疏点式水听器阵列，适合发现未知位置目标。

### 限制

- **不是直接声压计**：绝对压力灵敏度和幅值可比性必须通过标准声源、共址水听器等方式校准。
- **方向性与标距效应**：响应受入射角、波长和 gauge length 影响；相邻 channel spacing 不等于独立传感器间距。
- **流噪突出**：拖曳时的涡激振动（strumming）、尾摆和湍流边界层压力波动都会造成强低频噪声。
- **相干衰落与坏道**：瑞利散射随机性可能产生低信噪通道，需要质量控制和掩码机制。
- **带宽—量程—数据率折衷**：长缆、高采样率、密空间采样不能无限同时提高；数据量很大。
- **小规模系统未必便宜**：感知缆可以简单，但 interrogator 仍是昂贵且关键的系统组件。

## 5. 对 Deepship 的影响

当前 Deepship 基线使用的是单通道水听器 WAV：16 kHz、3 秒、64×94 log-Mel 特征。DAS 的自然输入则是：

\[
X(\text{distance},\ \text{time})
\]

或时频张量：

\[
X(\text{channel},\ \text{frequency},\ \text{time})
\]

因此，不能直接把 DeepShip 的 16 kHz 音频当作真实 DAS 数据。公开拖曳 DAS 海试主要服务主动海洋地震，实测有效反射频段约 20–150 Hz；相比之下，DeepShip 16 kHz 音频可含至 8 kHz 的信息。除了频带差异，还存在测量物理量、方向性、流噪和空间结构的差异。

## 6. 建议的数据与模型路线

### 数据采集与标定

1. 进行 DAS 拖缆与校准水听器的**共址配对采集**。
2. 保存 DAS 相位/应变/应变率及原始 I/Q（如可获得），并保存输出单位。
3. 每条数据记录以下元数据：缆型、增敏结构、标距、channel spacing、采样率、interrogator 设置、拖速、缆深、航迹、目标方位/距离、海况、流噪指标和船只 MMSI。
4. 用已知声源及共址水听器估计或学习局部 \(H_{p\rightarrow\epsilon}(f,\theta,z)\)。

### 模型设计

建议将现有单谱图分类器扩展为双分支模型：

```text
DAS channel × time
  ├─ 局部通道时频分支：log-Mel / MA-CNN-A 类特征，学习船噪频谱
  ├─ 空间—时间分支：2D CNN 或 Transformer，学习跨通道到达斜率、相干性与阵列结构
  └─ 融合层：结合通道质量掩码、缆型/标距等条件信息，输出船型或任务标签
```

训练增强应包含随机通道增益、频率传递函数、相位噪声、坏道遮挡、流噪、标距变化及方位变化，而不只使用普通音频增强。

### 迁移步骤

1. **带宽匹配基线**：将 DeepShip 音频滤波、重采样到目标 DAS 实际频带，验证低频声学类别信息是否仍然存在。
2. **域模拟预训练**：可用 \(X_\mathrm{DAS}(f)=H(f,\theta,z)X_\mathrm{hydrophone}(f)+N(f)\) 生成受控模拟数据；它只能用作预训练或消融，不能替代真实 DAS 数据。
3. **真实配对数据微调**：用共址 DAS—水听器数据完成监督或半监督适配。
4. **自监督预训练**：在大量无标签 DAS 数据上做 masked modeling、对比学习或跨通道一致性学习，再进行船型分类微调。

### 泛化评测

除了已有的录音级/船名级隔离，还应按以下实体隔离：

- 物理船只 ID（优先 MMSI/IMO）；
- 航次或采集 campaign；
- 缆型、增敏结构和标距；
- 拖速、深度、海况和背景噪声；
- 目标距离与方位区间。

目标是评估网络是否真正学到船源特征，而不是记住某一条缆、某次航行、某种流噪或特定方向的传递函数。

## 7. 可直接引用的表述

> 拖曳式 DAS 将光纤转化为高密度分布式应变阵列，可为水下目标探测提供传统水听器阵列难以获得的连续空间观测。然而，DAS 对光纤轴向应变而非声压直接响应，其幅频特性受缆体结构、标距、入射方向、流噪及部署耦合共同影响。因而，面向 DAS 船只识别的泛化网络应将船源声学特征与“缆—环境—方位”传递函数分离建模，并通过配对标定、空间—时间特征学习和跨航次/跨缆型隔离评测验证泛化能力。

## 8. 参考资料

1. He et al., *Marine Seismic Exploration with Distributed Acoustic Sensing*, Engineering, 2026. <https://www.engineering.org.cn/engi/EN/PDF/10.1016/j.eng.2025.04.007>
2. Matsumoto et al., *Detection of hydroacoustic signals on a fiber-optic submarine cable*, Scientific Reports, 2021. <https://www.nature.com/articles/s41598-021-82093-8>
3. He & Liu, *Optical Fiber Distributed Acoustic Sensors: A Review*, Journal of Lightwave Technology, 2021. <https://opg.optica.org/jlt/abstract.cfm?URI=jlt-39-12-3671>
4. Näsholm et al., *Array Signal Processing on DAS Data: Directivity Effects in Slowness Space*, JGR Solid Earth, 2022. <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB023587>
5. Bouffaut et al., *Eavesdropping at the Speed of Light: DAS of Baleen Whales in the Arctic*, Frontiers in Marine Science, 2022. <https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.901348/full>
6. Zhu et al., *Distributed fiber optic hydrophone based on backscattering-enhanced DAS*, SPIE, 2025. <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13542/135422J/Distributed-fiber-optic-hydrophone-based-on-backscattering-enhanced-DAS/10.1117/12.3055906.full>
