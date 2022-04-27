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



