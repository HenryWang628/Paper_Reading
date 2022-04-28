##### xDeepFM & DCNv2

- ##### xDeepFM

  保留了Deep和Cross的基本结构， 着重对于Cross部分进行了结构上的修改。对于输入特征embedding，全部拼接起来形成矩阵 $X^0 ∈ R^{m×D}$ 其中 $m$ 表示特征个数， $D$  表示 embedding的维度， 第k-1层的feature map记作 $X^{k-1}$ ,但是它的第一维不是m，暂时记作 $H_{k-1}$。通过k-1层推导至第k层的操作可以表示为：

  ![[公式]](https://www.zhihu.com/equation?tex=X%5Ek%5Bh%2C%3A%5D%3D%5Csum_%7Bi%3D1%7D%5E%7BH_%7Bk-1%7D%7D%5Csum_%7Bj%3D1%7D%5E%7Bm%7D+W_%7Bi%2Cj%7D%5E%7Bk%2Ch%7D%28X%5E%7Bk-1%7D%5Bi%2C%3A%5D%5Codot+X%5E0%5Bj%2C%3A%5D%29)

  $X^0[j,:]$就表示取出这个矩阵的第$j$行。 $j, i, h$ 分别表示输入矩阵，第$k-1$层 feature map和第 $k$  层feature map的**行号索引**。那么新的feature map中的每一行，都是先让上一层的每一行，和每一个输入embedding做element-wise交叉，再用一套独有的 ![[公式]](https://www.zhihu.com/equation?tex=W) 做变换后加起来融合的。

  ![img](https://pic2.zhimg.com/80/v2-0297836835eabb76ad8fdc1c22ad4055_720w.jpg)

  在图中，每一个圆都表示一个元素。红色方框框起来的部分是一个embedding。灰色是原始特征 ![[公式]](https://www.zhihu.com/equation?tex=X%5E0) ，蓝色和黄色分别是 ![[公式]](https://www.zhihu.com/equation?tex=X%5E%7Bk-1%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=X%5Ek) 。我们把灰色和蓝色的每一行分别交叉后，乘以对应的 ![[公式]](https://www.zhihu.com/equation?tex=W) ，加起来得到最终 ![[公式]](https://www.zhihu.com/equation?tex=X%5Ek) 的第一行。

- 进一步解释：

  - xDeepFM 是vector-wise的交叉， DCN本质是bit-wise的交叉
  - 把每一层的feature map都直连在输出上，是为了高阶和低阶的都传递到输出，使得最后包含各阶交叉信息
  - 名字中的compressed（压缩）主要体现在哪里？主要是 $k-1$ 层，与原始embedding作用后，通过加权求和，最终剩下有限个embedding拼接的矩阵。交叉的结果并不会无限膨胀。**因此求和这里其实就是做了压缩**。比如我们可以令每一层 $H$ 都相同，而且是一个比较小的数字。



##### DCN-V2:

DCN V1的结构如图：

![img](https://pic1.zhimg.com/80/v2-063d50bfddc03e88ef78138b6c3a334c_720w.jpg)

它的核心表达式为：![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%2B1%29%7D%3Dx_0x_%7B%28l%29%7D%5ETw%2Bb%2Bx_%7B%28l%29%7D)

而V2的结构为：

![img](https://pic1.zhimg.com/80/v2-ad511d769a222bf1a12301942dcfc00c_720w.jpg)

它的表达式可以写为： ![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%2B1%29%7D%3Dx_0%5Codot+%28W_lx_%7B%28l%29%7D%2Bb_l%29%2Bx_%7B%28l%29%7D)

这里的定义和上一讲保持一致。可以看出，**最大的变化是将原来的向量 ![[公式]](https://www.zhihu.com/equation?tex=w) 变成了矩阵**。而这一个改动就解决了前面最大的问题。**一个矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W) 拥有足够多的参数来保留高阶交叉信息，或者挑选需要的交叉结果**。在现在这个结构下，就可以还原出HOFM的形式了（这里留作思考题，读者可以自行推一下），因此这个工作也兑现了真正的高阶交叉。

要注意的一个**DCN-V2和xDeepFM的很大区别是，DCN-V2仍然不是vector-wise的操作**。根源在于，DCN-V2把所有特征的embedding拼起来一起输入网络，所以在 ![[公式]](https://www.zhihu.com/equation?tex=W_l) 那里无法保持同一个特征的embedding同进退，同一段embedding自己内部也存在交叉。而模型的结构又要求 ![[公式]](https://www.zhihu.com/equation?tex=W_l) 是个方形矩阵，这样参数量就会非常大。因此作者引入了低秩分解来处理，即把 ![[公式]](https://www.zhihu.com/equation?tex=W_l) 变成两个小矩阵的乘，即 ![[公式]](https://www.zhihu.com/equation?tex=U%2C+V%5Cin+R%5E%7Bd%5Ctimes+r%7D%2Cr%5Cle+d%2F2) , ![[公式]](https://www.zhihu.com/equation?tex=UV%5ET%3DW) ：

![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%2B1%29%7D%3Dx_0%5Codot+%28U_l%28V_l%5ETx_%7B%28l%29%7D%29%2Bb_l%29%2Bx_%7B%28l%29%7D) 。
