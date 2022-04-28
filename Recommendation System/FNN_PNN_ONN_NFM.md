##### FNN->PNN->ONN->NFM

*->感觉FM和DNN没直接关系*

->*FM可以和DNN相互作用，但有点别扭*

->*FM可以以比较优雅的方式融合进DNN*

->*对FM的内核重新思考，保留最关键的部分，同时保留更多的可能*。

##### FNN

![FNN](https://pic3.zhimg.com/v2-dd02c9900bb39dd3ef4018cf383ac86e_r.jpg)

- 分阶段训练：FM和DNN
- FM对FNN是援助关系：
  - 训练FM并使用这些embedding初始化从one-hot到第一层Dense层的权重矩阵W
  - 使用FM给DNN最难学习的部分打基础

##### PNN：

![PNN](https://pic1.zhimg.com/v2-43935307f327b7c6a6dbab355b103f3c_r.jpg)

- input 部分：
  - 使用了embedding layer, 先从One-hot生成embedding，然后插入FM
  - $f$ 表示dense embedding
- 特征交叉部分：
  - 左边的1和 $f$ 生成一阶部分$z$,  $f$ 两两进行作用product生成 $p$，下面的激活元是通过$z$和 $p$分别进行线性映射，然后加上偏置，合在一起得到。
  - 相互作用包括内积和外积，对应模型为IPNN和OPNN
  - 外积 ![[公式]](https://www.zhihu.com/equation?tex=g%28f_i%2Cf_j%29%3Df_if_j%5ET) 这样得到的结果是一个矩阵，那么下面映射到L1那里就需要一个3维的矩阵来做转换。如果每次都先算完两两的外积然后再按照FM求和，复杂度是有点爆炸的。这篇文章在这里做了一个近似：先计算所有 ![[公式]](https://www.zhihu.com/equation?tex=f) 的和，再做外积，即 ![[公式]](https://www.zhihu.com/equation?tex=p%3Df_%5Csum+f_%5Csum%5ET) 。这样先计算求和的复杂度不高，最后只需要做一次点积就行了。

##### ONN

![img](https://pic4.zhimg.com/80/v2-cf87f1970cee2b2f10fdec32055340c3_720w.jpg)



ONN所定义的operation-aware，其实本质也是像FFM一样，允许更大的自由度，只是划分方式，在FFM里面叫做field，在这里叫做operation而已。ONN的结构如下，可以看做是在PNN的基础上加了分组的操作：

实践中选择哪些特征交叉到一起是比较有意思的（可能也是大多数算法工程师真正日常在做的工作）。这里介绍一些个人的经验：

- 类型相似的特征，可以放在同一个field里面，比如user的城市，年龄，性别，这些都是静态的，短时间内不会变化，就可以放在一起
- field之间尽量有差别，比如第一个以用户静态特征为主题，第二个field可以以用户兴趣标签之类的为主题，第三个可以选到作者侧上
- 交叉的时候倾向于user交叉item。 往往来说，item或者user自己内部的交叉收益不太大，而user和item的共现（co-occurrence）更加重要

##### NFM

- 只做element-wise的相乘，不求和：

  ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29%3D%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di%2B1%7D%5Enx_ix_j+v_i%5Codot+v_j)

- 其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Codot) 表示element-wise乘法。在这里的原型还是按照原始FM来的，把所有的交叉都表示到element-wise乘法这一步就停止，保持二阶信息作为输入送进DNN中，我们都知道DNN具有很好的非线性，二阶信息叠加高度非线性，那我们其实是有理由展望一下更高阶的交叉**可能**存在。-
- 其实在NFM这里就隐隐透露出可以做更高阶的交叉的影子了，但是为什么我们只强调是**可能**存在呢，因为MLP不会替你做交叉，而且这里的交叉还是不如DCN里面形式上那么明显。NFM的主要立论点是要扩展FM的上限，强化FM的能力，其实主要目标不在高阶交叉上，而DCN这一类方法就完全是奔着高阶交叉去的了。



##### 总结来说：

总共三种信息：

- 单独的embeddin：记为 ![[公式]](https://www.zhihu.com/equation?tex=v_i) 

- element-wise乘后的二阶embedding：![[公式]](https://www.zhihu.com/equation?tex=v_i%5Codot+v_j) 

- 点积结果 ![[公式]](https://www.zhihu.com/equation?tex=%3Cv_i%2C+v_j%3E) 

可选的操作方案：

- 对 ![[公式]](https://www.zhihu.com/equation?tex=v_i) ，可以让它参与element-wise乘法或者点积的运算，但是同时可以share一份单独供给DNN做输入；
- element-wise乘的结果 ![[公式]](https://www.zhihu.com/equation?tex=v_i%5Codot+v_j) ，可以独立给DNN输入，也可以共享一份得到点积的结果；
- 点积结果可和DNN的输出加起来整体作为输出，也可单独拉一个loss，仅仅是为了辅助embedding的训练。





 
