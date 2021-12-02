## 传统推荐模型

#### 1.1 关系图

![image-20211102161949341](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102161949341.png)

#### 1.2 协同过滤（CF）

- 基本思想：人以类聚，物以群分

- 目标场景：

- 相似度计算方法：

  - 杰拉德（Jaccard）相似度：

    ​                                            $J(A,B)=\frac{|A\bigcap B|}{|A\bigcup B|}$

  - 余弦相似度：计算用户向量i和j之间的夹角大小，夹角越小，相似度越大。

    ​											$sim(i,j)=cos(i,j)=\frac{i\cdot j}{||i||\cdot ||j||}$

    - 较为常用，但存在局限性：

    - 评分不规范，如有的用户天生喜欢打高分/低分

  ```python
  from sklearn.metrics.pairwise import cosine_simlarity
  i = [1, 0, 0, 0]
  j = [1, 0.5, 0.5, 0]
  cosine_similarity([i,j])
  ```

  

  - 皮尔逊相关系数：对于用户评分偏置的情况，可以考虑使用Pearson相关系数

    ​									<img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102163828080.png" alt="image-20211102163828080" style="zoom:67%;" />                             $ R_i = [R_{i1} \ R_{i2} ... \ R_{iN}]$,  $R_j = [R_{j1} \ R_{j2} ... \ R_{jN}]$

  ```python
  from scipy.stats import pearsonr
  
  i = [1, 0, 0, 0]
  j = [1, 0.5, 0.5, 0]
  pearsonr(i,j)
  ```

  - 其他
    - 欧式距离
    - 曼哈顿距离
    - 马氏距离

  - 欧式距离 vs 余弦相似度

    - 欧式距离体现数值上的绝对差异，余弦距离体现方向上的相对差异
    - 欧式距离强调绝对数值，余弦相似强调夹角
    - 例子：

    | 场景                                                         | 选择         |
    | ------------------------------------------------------------ | ------------ |
    | 统计两部剧用户观看行为，用户A观看向量（0,1）用户B观看向量（1,0）。分析两用户对不同视频的喜好。（此时余弦距离很大，而欧式距离很小） | 选择余弦距离 |
    | 分析用户活跃度。以登录次数和平均观看时长为特征。用户A（1,10）用户B（1,100）。（此时余弦距离会很近，但是欧式距离很远） | 选择欧氏距离 |

- ##### 基于User的协同过滤（UserCF）：计算用户之间的相似度

  - ###### 步骤一：计算Alice和其他用户相似度

  <img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102165422212.png" alt="image-20211102165422212" style="zoom:50%;" />

  <img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102165456316.png" alt="image-20211102165456316" style="zoom: 67%;" />

  用户向量Alice, user1, user2, user3, user4

  (1) 余弦相似性：sim(Alice, user1) = cos(Alice, user1) = $\frac{15+3+8+12}{\sqrt{25+9+16+16}*\sqrt{9+1+4+9}}=0.975$

  (2) Pearson相关系数 Alice_ave = 4  user1_ace =2.25

  ​							sim(Alice, user1) = 0.852 

  ```python
  from sklearn.metrics.pairwise import cosine_similarity
  users = np.array([[5, 3, 4, 4],[3, 1, 2, 3],[4, 3, 4, 3],[3, 3, 1, 5],[1, 5, 5, 2]])
  cosine_similarity(users)
  np.corrcoef(users)
  ```

  - ###### 步骤二：预测得分

    - 方式一：加权平均	

      用户u对于商品p的评分：							

      ​																					$R_{u,p} = \frac{\sum_{s∈S}w_{u,s}R_{s,p}}{\sum_{s∈S}w_{u,s}}$

      $w_{u,s}$表示用户u和s之间的相似度

      （部分用户喜欢打高分，有的喜欢打低分，容易不客观）

    - 方式二：

      ​																		$P_{i,j} = \overline{R_i}+\frac{\sum^n_{k=1}(S_{i,k}(R_{k,j}-\overline{R_k}))}{\sum^n_{k=1}S_{i,k}}$

  ​			这里$P_{i,j}$表示用户$i$对商品$j$的评分，S表示相似度，R同样表示评分

  ​															$P_{Alice,物品5} = \overline{R_{Alice}}+\frac{\sum^2_{k=1}(S_{Alice,userk}(R_{userk,物品5}-\overline{R_{userk}}))}{\sum^2_{k=1}S_{Alice,userk}}$=4.87

  - ###### 步骤三：基于用户评分进行推荐

    设定阈值，超过阈值可以推荐给用户

  ##### UserCF的缺点：

  - 数据稀疏性：商品多，用户之间买的重叠性比较低，导致难以找到一个用户的邻居（偏好相似用户）。即使找到了也准确性不高，所以UserCF不适用于正反馈获取困难的应用场景

    （如酒店预订，大件商品购买的低频应用）

  - 用户相似度矩阵维护难度大：

    - 互联网场景中用户数一般远大于无评书，维护用户相似度矩阵难度大

    - 基于用户协同过滤需要维护用户相似度矩阵以便快速找出Top n的相似用户，该矩阵的存储开销巨大，不适用于用户数据量大的情况使用

  ##### 使用场景：

  - 适用于用户少，物品多，时效性强的场合（如新闻推荐场景）

- ##### 基于Item的协同过滤（ItemCF）：计算物品之间的相似度（电商早期使用）

  <img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102172957997.png" alt="image-20211102172957997" style="zoom:67%;" />

  计算过程类似，但是是计算列向量之间的相似度，即商品向量之间的相似度

  ​																	$P_{Alice,物品5} = \overline{R_{物品5}}+\frac{\sum^2_{k=1}(S_{物品5,物品k}(R_{Alice,物品k}-\overline{R_{物品k}}))}{\sum^2_{k=1}S_{物品k,物品5}}$

  ​																				= $\frac{13}{4}+\frac{0.97*(5-3.2)+0.58*(4-3.4)}{0.97+0.58}=4.6$

  ##### ItemCF优点：

  - Item-based的预测效果更好，余弦计算好物品相似度，在线预测性能更好

    （物品增长速度较慢，较稳定，能在较长时间内维护较稳定的相似度）

  ##### ItemCF缺点：

  - 稀疏性
  - 相似度维护难度大（但相对UserCF可能较小）

  ##### 适用场景：

  - 兴趣变化较为稳定的应用。更接近个性化的推荐

  - 用户数量远大于商品数目，用户兴趣固定持久，商品更新速度不是太快

    （推荐艺术品、音乐、电影等）

- ##### UserCF和ItemCF的优缺点对比

  |          | UserCF                                                       | ItemCF                                                       |
  | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 性能     | 适用于用户较少场合，如果用户很多，计算用户相似矩阵代价大     | 适用于物品数明显校友用户数的场合，如果物品很多，计算武平相似度矩阵代价也很大 |
  | 领域     | 时效性强，用户个性化兴趣不太明显的领域（强调人与人之间的共性，周围人都在看） | 长尾物品丰富，用户个性化需求强烈的领域（强调个性）           |
  | 实时性   | 用户有新行为，不一定造成推荐结果的立即变换                   | 用户有新行为，一定会导致推荐结果实时变化                     |
  | 冷启动   | 在新用户对很少的物品产生行为以后，不能立即对他进行个性化推荐，因为用户相似度表每隔一段时间离线计算（需要更新相似用户，才能做出准确推荐） | 新用户只要对一个物品产生行为，就可以给他推荐和该物品相关的其他物品 |
  | 新物品   | 新物品上线一段时间，一旦有用户对物品产生行为，就可以将新物品推荐给对它产生行为的用户兴趣相似的其他用户 | 没有办法在不离线更新物品相似度表的情况下，将新物品推荐给新用户 |
  | 推荐理由 | 难提供令用户信服的推荐解释                                   | 利用用户的历史行为给用户做推荐解释，可以令用户比较信服       |

  | 共同缺点                                                     |
  | :----------------------------------------------------------- |
  | 不能彻底解决数据稀疏性的问题                                 |
  | 泛化能力弱：热门商品具有很强的头部效应，容易和大量物品产生相似，而尾部物品由于特征向量系数，很少被推荐（为有效解决头部效应，矩阵分解技术被提出） |
  | 无法利用更多信息，如用户和物品本身的特征                     |

#### 1.3 MF矩阵分解 -- SVD、LFM、RSVD、SVD++（Matrix Factorization）

- ##### 针对问题：

  ​					协同过滤处理稀疏矩阵的能力较弱

  ​					协同过滤中，相似度矩阵维护难度大

- ##### 解决思路：

  

  ![image-20211102211501857](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102211501857.png)

![image-20211102211614854](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102211614854.png)

- 

  - 隐含特征是不可解释的，需要模型自己学习

  - k的大小决定隐向量表达能力强弱，k越大表达能力越强，用户兴趣和物品分类具体
  - 通过用户矩阵和物品矩阵预测评分计算公式：
  
    ​								$Preference(n,i)=r_{ui} = \sum^F_{f=1}p_{u,k}q_{k,i}$    (对应向量内积)
  
- ##### MF方式

  - ###### 特征值分解

    - 特征值，特征向量 ： $Av = \lambda v$

      ​				$v$是特征向量，$\lambda$是特征向量

    - 特征值分解： $A = Q\sum Q^{-1}$ 

      ​					$Q$代表矩阵A的特征向量构成的矩阵

      ​					$\sum$是对角阵，对角线的元素是特征值

  - ###### 奇异值分解（SVD）：

    - 定义：  $A = U\sum V^T$  

      其中	A是实矩阵，$UU^T=I, VV^T=I$

      $\sum$是对角矩阵，对角线元素非负且降序排列

      ![image-20211102213539391](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102213539391.png)

    - 计算步骤

         A是一个m*n的实矩阵

      1. 构造n阶实对称矩阵 $W = A^TA$

      2. 计算W的特征值与特征向量

         ![image-20211102220119826](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102220119826.png)

      3. 求得n阶正交矩阵V

         ![image-20211102220133591](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102220133591.png)

      4. 求得m*n对角矩阵

         ![image-20211102220150005](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102220150005.png)

      5. 求得m阶正交矩阵U（求得上述）

         - 求U1

           <img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102220815044.png" alt="image-20211102220815044" style="zoom: 80%;" />

         - 求U2

           <img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102220846565.png" alt="image-20211102220846565" style="zoom: 80%;" />

         - 得到U=[U1, U2]

    - 缺点：

      传统SVD分解会要求原始矩阵是稠密的，所以我们需要对缺失进行填补，空间复杂度非常高，基本无法解决大规模稀疏矩阵的矩阵分解问题

  - ###### Basic SVD（LFM，Funk SVD）

    - 将矩阵分解问题转化为**最优化问题**，通过梯度下降进行优化

    - 预测函数：

      ​							<img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102224155367.png" alt="image-20211102224155367" style="zoom:67%;" />

    - 损失函数（误差平方和）：

      式子2便于优化

      ![image-20211102221418024](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102221418024.png)

    - 步骤：
      1. 首先先初始化这两个矩阵
      2. 把用户评分矩阵里面已经评过分的那些样本当做训练集的label， 把对应的用户和物品的隐向量当做features， 这样就会得到(features, label)相当于训练集
      3. 通过两个隐向量乘积得到预测值pred
      4. 根据label和pred计算损失
      5. 然后反向传播， 通过梯度下降的方式，更新两个隐向量的值
      6. 未评过分的那些样本当做测试集， 通过两个隐向量就可以得到测试集的label值
      7. 这样就填充完了矩阵， 下一步就可以进行推荐了

  - ###### RSVD

    在Basic SVD基础上，加入正则化参数（惩罚项）

    预测函数：

<img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102224202728.png" alt="image-20211102224202728" style="zoom:67%;" />

​				目标函数：

![image-20211102224009842](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102224009842.png)

- 

  - ###### 改进（LFM）：

    Netflix提出另外一种LFM，在原有基础上加偏置项，消除用户和物品打分的偏差

    原因：不同用户打分体系不同，不同物品衡量标准有区别，导致评分偏差

  <img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211102224746963.png" alt="image-20211102224746963" style="zoom:67%;" />

  ​		目标函数：

  <img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211103095822683.png" alt="image-20211103095822683" style="zoom:80%;" />

  - ###### SVD++

    改进方向：用户历史记录会对新评分产生影响（即物品间存在某些联系），交给模型学习

    ![image-20211103102425144](C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211103102425144.png)

  <img src="C:\Users\Huayra\AppData\Roaming\Typora\typora-user-images\image-20211103102506355.png" alt="image-20211103102506355" style="zoom: 50%;" />

