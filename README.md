# Paper_Reading

## Natural Language Processing
|Time|Model|Finish?|Link|
|:-:|:-:|:-:|:-:|
|2021-12-01|Transformer|✅|[Link](https://github.com/HenryWang628/Paper_Reading/tree/main/Transformer)|
|2021-12-01|BERT|✅|[Link](https://github.com/HenryWang628/Paper_Reading/tree/main/BERT)|

## General Models
|Time|Model|Finish?|Link|
|:-:|:-:|:-:|:-:|
|2021-12-02|GCN|✅|[Link](https://distill.pub/2021/gnn-intro/) |

## Machine Learning Models
|Time|Model|Finish?|Link|
|:-:|:-:|:-:|:-:|
|2021-12-02|XGboost|✅|[Link](https://github.com/HenryWang628/Paper_Reading/blob/main/Machine%20Learning/XGBoost.pdf) |
|2022-1-08|GBDT|✅|[Link](https://www.cnblogs.com/pinard/p/6140514.html) |

## Recommendation system
![image](https://github.com/HenryWang628/Paper_Reading/blob/main/pic/RS.JPG?raw=true)
|Time|Model|Finish?|Link|
|:--------:|:--------------:|:--:|:-:|
|2021-11-01|UserCF & ItemCF | ✅ |[基本概念&数学推导](https://github.com/HenryWang628/Paper_Reading/blob/main/Recommendation%20System/UserCF%26ItemCF.pdf)|
|2021-11-15|     MF         | ✅ |[基本概念&数学推导](https://github.com/HenryWang628/Paper_Reading/blob/main/Recommendation%20System/MF--%20SVD%E3%80%81LFM%E3%80%81RSVD%E3%80%81SVD%2B%2B%EF%BC%88Matrix%20Factorization%EF%BC%89.pdf)|
|2021-12-04|     LR         | ✅ |[基本概念&数学推导](https://github.com/HenryWang628/Paper_Reading/blob/main/Recommendation%20System/LR.pdf) ,[推导2](https://github.com/HenryWang628/Paper_Reading/blob/main/Recommendation%20System/LR.PNG)|
|2021-12-02|     POLY2      | ✅ |[基本概念](https://github.com/HenryWang628/Paper_Reading/blob/main/Recommendation%20System/POLY2.md), [数学推导](https://github.com/HenryWang628/Paper_Reading/blob/main/Recommendation%20System/POLY2.pdf)|
|2021-12-04|     FM         | ✅ |[基本概念&数学推导](https://github.com/HenryWang628/Paper_Reading/blob/main/Recommendation%20System/FM.pdf)|
|2021-12-04|    FMM         | ✅ |[基本概念&数学推导](https://github.com/HenryWang628/Paper_Reading/blob/main/Recommendation%20System/FFM.pdf)|
|          |    GBDT+LR     |    |[基本概念&数学推导]()|
|          |   Wide&Deep    |    |[基本概念&数学推导]()|
|          |    DeepFM      |    |[基本概念&数学推导]()|



### Recommendation system summary
### 传统模型
| 模型                | 基本原理                                                     | 特点                                                   | 局限                                                         |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ |
| 协同过滤（CF）      | 根据用户的行为历史生成用户-物品共现矩阵                      | 原理简单                                               | 泛化能弱，处理稀疏矩阵的能力差，推理结果的头部效应较为明显   |
| 矩阵分解（MF）      | 将协同过滤算法中的共现矩阵分解为用户矩阵和物品矩阵，利用用户隐向量和物品隐向量的内积进行排序并推荐 | 能融合多种类型的不同特征                               | 除了用户历史行为数据，难以利用其它用户、物品特征及上下文特征 |
| 逻辑回归            | 将推荐问题转化为类似CTR预估的二分类问题，将用户、物品、上下文等不同特征转化为特征向量、输入逻辑回归模型得到CTR，再按照预估CTR进行排序并推荐 | 能够融合多种类型的不同特征                             | 模型不具备特征组合的能力、                                   |
| 因子分解机（FM）    | 在逻辑回归的基础上，在模型中加入二阶特征交叉部分，为每一维特征训练得到相应特征隐向量，通过隐向量间的内积运算得到特征权重 | 相比逻辑回归，具备了二阶特征交叉能力，模型表达能力增强 | 由于组合爆炸问题的限制，模型不易拓展到三阶特征交叉阶段       |
| 域因子分解机（FFM） | 在FM的基础上，加入特征域的概念，使得每个特征在不同域的特征交叉时采用不用的隐向量 | 相比FM，进一步加强特征交叉能力                         | 模型训练开销大                                               |
| GBDT-LR             | 利用GBDT进行自动化特征组合，使得原始特征向量转化为离散型特征向量，并输入逻辑回归模型，进行最终的CTR预估 | 特征工程模型化，使模型具备了更高阶特征组合的能力       | GBDT无法进行完全并行的训练，更新所需的训练时长较长           |
| LS-PLM              | 对样本进行分片，在每个分片内部构造逻辑回归模型，将每个样本的各“分片”概率与逻辑回归的得分进行加权平均，得到最终的预估值 | 模型类似三层神经网络，具备较强表达能力                 | 模型结构相比深度模型仍然较为简单，进一步提高空间             |

### 深度模型

| 模型          | 基本原理                                                     | 特点                                                         | 局限                                                     |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------- |
| AutoRec       | 基于自编码器，对用户或者物品进行编码，利用自编码器的泛化能力进行推荐 | 单隐层神经网络结构简单，课实现快速训练和部署                 | 表达能力差                                               |
| Deep Crossing | 利用“Embedding+多隐层+输出层”的经典模型，预完成特征的自动深度交叉 | 经典的深度学习推荐模型框架                                   | 利用全理解隐层进行特征交叉，针对性不强                   |
| NueralCF      | 将传统的矩阵分解（MF）中用户向量和物品向量的点积操作，换成神经网络代替的互操作 | 表达能力加强版的矩阵分解                                     | 只使用了用户和物品的id特征，没有加入其它特征             |
| PNN           | 针对不同特征域之间的交叉操作，定义“内积”，“外积”等多种积操作 | 在深度学习框架上模型对提高特征交叉能力                       | “外积”操作进行了近似化，一定程度上影响力其表达能力       |
| Wide&Deep     | 利用Wide部分加强模型的“记忆能力”，利用“Deep”部分加强模型泛化能力 | 开创组合模型的构造方法，对深度学习推荐模型产生重大影响       | Wide部分需要认购进行特征组合的筛选                       |
| Deep&Cross    | 用Cross网络替代Wide&Deeo中的Wide部分                         | 解决Wide&Deep中人工组合特征的问题                            | Cross网络复杂度较高                                      |
| FNN           | 利用FM的参数来初始化深度神经网络的Embedding参数              | 利用FM初始化参数，使得整个网络的收敛速度加快                 | 模型简单，没有针对性特征交叉层                           |
| DeepFM        | 在Wide&Deep模型的基础上，用FM替代原理线性Wide部分            | 加强Wide部分特征交叉能力                                     | 与经典的Wide&Deep相比，结构差别不大                      |
| NFM           | 用神经网络代替FM中二阶隐向量交叉的操作                       | 相比FM，NFM的表达能力更强                                    | 与PNN模型结构非常相似                                    |
| AFM           | 在FM基础上，在二阶隐向量交叉的基础上，对每个交叉结果加入了注意力得分，并使用注意力网络学习注意力得分 | 不同交叉特征的重要性不同                                     | 注意力网络训练过程比较复杂                               |
| DIN           | 在传统深度推荐模型的基础上引入了注意力机制，利用用户行为历史物品和目标广告物品的相关性计算注意力得分 | 根据目标广告物品的不同，进行更有针对性的推荐                 | 并没有充分利用历史行为以外的其他特征                     |
| DIEN          | 将序列模型与深度学习推荐模型的结合，使用序列模型模拟用户的兴趣金华过程 | 序列模型增强了系统对用户兴趣变迁的表达能力，使推荐系统开始考虑时间相关的行为序列中包含的有价值信息 | 序列模型训练复杂，线上服务延迟较长，需要进行工程上的优化 |
| DRN           | 强化学习应用于推荐系统，进行推荐系统线上试试学习和更新       | 模型对数据实时性的利用能力大大加强                           | 线上部分较为复杂，工程实现难度大                         |
