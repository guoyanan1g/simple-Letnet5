import torch
from torch import nn
from torch.nn import functional as F


#input : [b,3,32,32]
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        #卷积单元
        self.ConvUnit=nn.Sequential(
            nn.Conv2d(3,6,5,stride=1,padding=0),#[b,6, ,]
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6,16,kernel_size=6,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
        )
        #线性单元
        self.LinUnit=nn.Sequential(
            #送入前要将卷积层送来的tensor打平成一维，也就是[b,C*H*W]的形式
            nn.Linear(16*4*4,120),#不知道线性层的input是什么结构时，可以先测试了再填
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
        #测试卷积层输出的tensor结构
        tmp=torch.rand(10,3,32,32)
        out=self.ConvUnit(tmp)
        print(out.shape) #[10,16,4,4]

    def forward(self,input):
        """
        :param input: [b,3,32,32]
        :return: [b,10]
        """
        b=input.size(0)
        input=self.ConvUnit(input)
        input=input.view(b,input.size(1)*input.size(2)*input.size(3))#flatten
        logits=self.LinUnit(input)
        #pred=F.softmax(self.LinUnit(input),dim=1)
        #loss=nn.CrossEntropyLoss()
        return logits

def main():
    net=Lenet5()
    tmp=torch.rand(2,3,32,32)
    print(net(tmp))
   # print(F.softmax(net(tmp)))

if __name__ =='__main__':
    main()