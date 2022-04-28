##### HOFM & DCN

- ##### HOFM (High Order Factorization Machine)

  高阶特征交叉是将所有embedding做element-wise乘积再求和，乘以所有的交叉特征。

  计算中存在大量冗余，通过递推可以利用动态规划简化计算

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%2Cj%2Ck...%5Cin+S%7D+sum%28v_i%5Codot+v_j%5Codot+v_k...%29%28x_ix_jx_k...%29)

- ##### DCN

  这路网络会对特征进行**任意有限阶**交叉：

  ![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%2B1%29%7D%3Dx_0x_%7B%28l%29%7D%5ETw%2Bb%2Bx_%7B%28l%29%7D)

  在这里： ![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%29%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%2B1%29%7D) 分别是DNN中，输入层和输出层的中间结果（向量）。 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 为原始输入的embedding的拼接，它会在每一个层都参与运算

  ![img](https://pic3.zhimg.com/80/v2-e443530db2bba578bb973bcdee54afda_720w.jpg)

  这里的 ![[公式]](https://www.zhihu.com/equation?tex=y) 对应上面的 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%2B1%29%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=x%E2%80%99) 对应公式中的 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%29%7D) 。之所以要写成上面公式的形式是因为这个公式的模块是可以层层堆叠的。

  那么DCN这样的设计是如何进行特征交叉的呢？设想在第一层，其实就是 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 先和 ![[公式]](https://www.zhihu.com/equation?tex=x_0%5ET) 做向量外积，得到一个矩阵，然后在矩阵中，每一个元素都是原先embedding中两个元素的乘：

  ![[公式]](https://www.zhihu.com/equation?tex=x_0x_0%5ET%3D%5Cbegin%7Bbmatrix%7D+x_0%5E0x_0%5E0+%26+x_0%5E0x_0%5E1+%26+%5Ccdots+%26+x_0%5E0x_0%5En%5C%5C++x_0%5E1x_0%5E0+%26+x_0%5E1x_0%5E1+%26+%5Ccdots+%26+x_0%5E0x_1%5En+%5C%5C+++%26+%26++++%5Cddots+++%26++%5C%5C+x_0%5Enx_0%5E0+%26+x_0%5Enx_0%5E1+%26+%5Ccdots+%26+x_0%5Enx_0%5En+%5Cend%7Bbmatrix%7D)

  当后面再乘以 ![[公式]](https://www.zhihu.com/equation?tex=w) 的时候，其实是让 ![[公式]](https://www.zhihu.com/equation?tex=w) 进行筛选，选出哪些交叉项留下，继续进行后面的运算。在第一层计算完毕之后，结果其实保留了一部分二阶的embedding元素交叉，那么再往下继续，就会有3阶，4阶... 一直到网络层数的阶层。这样，**只要我的网络有 ![[公式]](https://www.zhihu.com/equation?tex=n) 层，我就能让输出带有 ![[公式]](https://www.zhihu.com/equation?tex=n) 阶的交叉**。下面的图展示了这个过程， ![[公式]](https://www.zhihu.com/equation?tex=x_0) 出现在每一层的中间，只要产生了新的中间输出，就要和 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 作用来得到下一层的结果：

  ![img](https://pic1.zhimg.com/80/v2-d49fea7d1f94dcbc6c6e0ae579cf1a94_720w.jpg)

在xDeepFM 这篇文章中相当于是做了一个归纳：

- **DCN的本质实际上是给 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 乘了一个系数**！

  重新考虑 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%2B1%29%7D%3Dx_0x_%7B%28l%29%7D%5ETw%2Bb%2Bx_%7B%28l%29%7D) 可以重新写作：

​														![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28l%2B1%29%7D%3Dx_0%28x_%7B%28l%29%7D%5ETw%2B1%29%2Bb) (这里是指从第一层开始递推)

结合上面的图，括号里面的两个向量分别是行向量和列向量，乘起来就是一个数字，也就是说，最后一层一层迭代完了，只得到一个 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 的倍数。你要说没有交叉吧，系数其实还是和 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 有关系的，你要说有交叉吧，又不是我们FM，PNN，ONN等等网络中讲得这么回事。DCN 并不是真正的高阶交叉。

