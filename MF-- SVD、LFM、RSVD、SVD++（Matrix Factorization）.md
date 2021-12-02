## MF矩阵分解 -- SVD、LFM、RSVD、SVD++（Matrix Factorization）

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

