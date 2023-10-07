# simple-Letnet5
A neural network based on Letnet5 model to fit and predict the CIFAR-10 datasets.
Without optimization of hyper-parameters.

CIFAR10数据集共有60000个样本，每个样本都是一张32*32像素的RGB图像（彩色图像），每个RGB图像又必定分为3个通道（R通道、G通道、B通道）。这60000个样本被分成了50000个训练样本和10000个测试样本。
不同类别的物体用不同的标签值，CIFAR10中有10类物体，标签值分别按照0~9来区分,他们分别是飞机（ airplane ）、汽车（ automobile ）、鸟（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、青蛙（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。

input: torch.Size([b,3,32,32]) b=BatchSize
output: torch.Size([b,10])
<br>

网络结构：
1.卷积层
2，avg池化层
3.卷积层
4.avg池化层
5.线性层
6.线性层
<br>
激活函数均为Relu




