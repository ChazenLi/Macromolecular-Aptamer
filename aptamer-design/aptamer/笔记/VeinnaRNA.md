# VeinnaRNA 可用程序以及功能
## 可用的程序
- **RNA2Dfold**：计算代表性样本结构的粗粒度能量景观
- **RNAaliduplex**：预测两个比对之间的保守RNA-RNA相互作用
- **RNAalifold**：计算一组比对RNA序列的二级结构
- **RNAcofold**：计算两个RNA分子的二级结构及其二聚化
- **RNAdistance**：计算RNA二级结构之间的距离
- **RNAduplex**：计算两个RNA链杂交后的结构
- **RNAeval**：评估给定二级结构的RNA序列的自由能
- **RNAfold**：计算RNA的最低自由能二级结构和配分函数
- **RNAheat**：计算RNA序列的比热（熔解曲线）
- **RNAinverse**：找到具有给定二级结构的RNA序列（序列设计）
- **RNALalifold**：计算一组比对RNA的局部稳定二级结构
- **RNALfold**：计算长RNA的局部稳定二级结构
- **RNApaln**：基于序列的碱基配对倾向进行RNA比对
- **RNApdist**：计算热力学RNA二级结构集合之间的距离
- **RNAparconv**：将ViennaRNA 1.8的能量参数文件转换为2.0格式
- **RNAPKplex**：预测包含假结的RNA二级结构
- **RNAplex**：找到查询RNA的靶标
- **RNAplfold**：计算局部稳定二级结构的平均配对概率
- **RNAplot**：用PostScript、SVG或GML绘制和标记RNA二级结构
- **RNApvmin**：找到可以进一步用来限制折叠的扰动能量向量
- **RNAsnoop**：找到查询H/ACA snoRNA的靶标
- **RNAsubopt**：计算RNA的次优二级结构
- **RNAup**：计算RNA-RNA相互作用的热力学
- **Kinfold**：模拟RNA序列折叠成二级结构的随机动力学
- **RNAforester 1**：通过森林比对比较RNA二级结构

## 可用的工具
- **b2ct**：将点括号表示法转换为Zukers mfold的`.ct`文件格式
- **b2mt.pl**：将点括号表示法转换为x y值
- **cmount.pl**：生成彩色山图
- **coloraln.pl**：为alirna.ps文件上色
- **colorrna.pl**：用可靠性注释为二级结构上色
- **ct2b.pl**：将Zukers mfold的`.ct`文件格式转换为点括号表示法
- **dpzoom.pl**：提取点图的一部分
- **mountain.pl**：生成山图
- **popt**：从subopt输出中提取Zuker的p-最优折叠
- **refold.pl**：使用共识结构作为约束重新折叠
- **relplot.pl**：向RNA二级结构图添加可靠性信息
- **rotate_ss.pl**：旋转RNA二级结构图的坐标
- **switch.pl**：描述具有两个几乎同样稳定结构的RNA序列
所有与ViennaRNA包一起提供的程序都提供了一些文档形式的“手册页”。在类似UNIX的环境中，成功安装ViennaRNA包后，可以使用`man`命令查看这些手册页。