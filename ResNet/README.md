# ResNet
ResNet本质并不是解决所谓过拟合的问题，因为随着网络越深，其模型在训练集和验证集的表现均发生了下降。（即训练误差和验证误差均上升）
本质应该是解决网络深度增加导致的网络退化/难优化的问题

参考：
https://zhuanlan.zhihu.com/p/268308900

### ResNet18 网络结构（Bottleneck只有ResNet50及以上才有）

![Resnet18](https://github.com/HenryWang628/Paper_Reading/blob/main/ResNet/Resnet18.PNG)
