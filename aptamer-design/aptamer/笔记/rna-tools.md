# 方法函数

## rna_pdb_tools.py
- **--get-rnapuzzle-ready**：将PDB文件格式转换为“RNA-Puzzle PDB格式”
- **--report**：获取报告
- **--renum-atoms**：重新编号原子，与`--get-seq`一起测试
- **--renum-residues-dirty**：重新编号残基（不干净）
- **--renumber-residues**：默认情况下为false
- **--delete-anisou**：删除包含ANISOU记录的文件，与`--inplace`一起使用
- **--split-alt-locations**：拆分备用位置
- **--clean**：获取干净的结构
- **--is-pdb**：检查文件是否为PDB格式
- **--is-nmr**：检查文件是否为NMR风格的多模型PDB
- **--un-nmr**：将NMR风格的多模型PDB文件拆分为单个模型【biopython】
- **--orgmode**：获取结构的org-mode格式
- **--get-chain**：获取链
- **--fetch**：从PDB数据库获取文件
- **--fetch-ba**：从PDB数据库获取生物组装体
- **--get-seq**：获取序列
- **--compact**：与`--get-seq`一起使用，获取紧凑视图
- **--get-ss**：获取二级结构
- **--rosetta2generic**：将ROSETTA样格式转换为通用PDB
- **--get-rnapuzzle-ready**
- **--collapsed-view**：折叠视图
- **--replace-hetatm**：将'HETATM'替换为'ATOM'【仅与`--get-rnapuzzle-ready`一起测试】
- **--mutate MUTATE**：突变残基
- **--edit EDIT**：编辑，例如`A:6>B:200`，`A:2-7>B:2-7`
- **--rename-chain RENAME_CHAIN**：重命名链
- **--swap-chains SWAP_CHAINS**：交换链
- **--replace-chain REPLACE_CHAIN**：替换链
- **--delete DELETE**：删除选定的片段，例如`A:10-16`，或者删除多个片段，例如`--delete 'A:1-25+30-57'`
- **--extract EXTRACT**：提取选定的片段，例如`A:10-16`，或者提取多个片段，例如`--extract 'A:1-25+30-57'`
- **--extract-chain EXTRACT_CHAIN**：提取链

## 序列分析
- **BlastPDB.py**：简单的Blast搜索
- **RfamSearch.py**：简单的Rfam搜索

## 二级结构分析
- **rna_secondary_structure_prediction.py**：二级结构预测方法的包装器，例如cyclefold、mcfold、ipknot、RNAsubopt、contextfold、centroid_fold，使用约束（如果适用）
- **rna_dot2ct.py**：将点表示法转换为ct表示法
- **二级结构格式转换工具**

## 三级结构比较
- **rna_calc_rmsd.py**：计算结构与目标的RMSD
- **rna_calc_evo_rmsd.py**：基于给定比对和选定残基计算结构之间的RMSD
- **rna_calc_inf.py**：基于ClaRNA的多进程（Python 2）
- **rna_clanstix.py**：基于成对结构相似性可视化RNA 3D结构的工具
- **rna_prediction_significance.py**：计算RNA三级结构预测的显著性

## 三级结构格式
- **diffpdb**：比较PDB文件的文本内容的简单工具
- **rna_pdb_merge_into_one.py**：将单个文件合并为NMR风格的多模型PDB文件

## 三级结构分析
- **clarna_app.py**：ClaRNA的包装器，见PyMOL4RNA，Python 2
- **rna_x3dna.py**：3dna的包装器，见PyMOL4RNA
- **ClashCalc.py**：简单的冲突得分计算器，使用于NPDock，需要BioPython

## 三级结构处理
- **rna_refinement.py**：QRNAS（快速精炼核酸）的包装器

## PyMOL4RNA
- **Undo ("Quick Save & Load") for PyMOL**：CTRL-S和CTRL-Z
- **PyMOL4Spliceosome**：链接到其自己的库
- **clarna()**：在PyMOL中直接对选定残基进行ClaRNA接触分类
- **x3dna()**：在PyMOL中直接对选定残基进行X3DNA接触分类
- **ss()**：获取选定对象的二级结构
- **sav <fn>**：在桌面上保存会话和PNG文件以展示会话
- **color structure domains according to pre-defined styles**：例如rp17()
- **PyMOL Preview Generator for OSX**

## SimRNA
- **rna_simrna_cluster.py**
- **rna_simrna_extract.py**
- **rna_simrna_get_data**
- **rna_simrna_lowest.py**
- **SimRNAweb: rna_simrnaweb_download_job.py**：下载模型文件，轨迹对于给定的SimRNAweb作业
- **rna_pdb_merge_structure_with_fragments.py**：在结构中插入片段，用于SimRNAweb服务器进行建模
- **rna_pdb_edit_occupancy_bfactor.py**：编辑PDB文件中的占有率或b因子
- **rna_pk_simrna_to_one_line.py**：将多行SimRNA二级结构格式转换为一行括号格式
- **rna_ss_pk_to_simrna.py**：执行相反操作，将带有假结的一行括号格式转换为多行SimRNA二级结构格式
- **见simrna_trajectory在Python Classes中**

## Rosetta
- **rna_rosetta_n.py**
- **rna_rosetta_check_progress.py**
- **rna_rosetta_min.py**
- **rna_rosetta_cluster.py**
- **rna_rosetta_extract_lowscore_decoys.py**
- **rna_rosetta_run.py**
- **rna_rosetta_head.py**

## RNA比对
- **get_seq()**：获取序列
- **get_ss()**：获取给定序列的二级结构
- **fetch()**：从Rfam获取比对
- **cmalign()**：将RNA序列比对到协方差模型（CM）
- **Rchie()**：绘制RNA二级结构的弧线图
- **find_core()**：在比对中找到分子的核心

## Python类
- **Seq.py**：序列处理，包括二级结构预测
- **SecondaryStructure.py::draw_ss()**
- **SecondaryStructure.py::parse_vienna_to_pairs()**
- **simrna_trajectory**

## 其他
- **rnakb_utils**：RNAkb相关工具
- **rnapuzzle_sender**：发送PDB文件到RNA-Puzzle组织者的脚本
- **rnashape2ascii**：将RNA形状数据转换为ASCII字符
- **cluster_load**：基于处理qstat查看集群负载的脚本