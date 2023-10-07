import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from lenet5 import Lenet5
from torch import  nn,optim
def main():
    batchsize=32
    device=torch.device("mps")# Mac M1

    cifar_train = datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_train=DataLoader(cifar_train,batch_size=batchsize,shuffle=True)

    cifar_test=datasets.CIFAR10('cifar',False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]))
    cifar_test=DataLoader(cifar_test,batch_size=batchsize,shuffle=True)
    """
    x,label=iter(cifar_train).__next__()
    print("x:",x.shape,"y:",label.shape) 
    ### X: torch.Size([32, 3, 32, 32]) y: torch.Size([32])
    """
    model=Lenet5().to(device)
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    criteon=nn.CrossEntropyLoss().to(device)
    #print(model)
    for epoch in range(100):
        model.train()
        for batchidx,(x,label) in enumerate(cifar_train):
            x,label=x.to(device),label.to(device)
            logits=model(x)
            loss=criteon(logits,label)#tensor scalar
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch," train cross entropy:",loss.item())
        #这里打印的是一次epoch中最后一个batch的交叉熵，所以参考意义并不大，最好直接看test的acc

        model.eval()
        with torch.no_grad(): #no need to back-propagate
            total_correct=0
            total=0
            for x,label in cifar_test:
                x,label=x.to(device),label.to(device)
                logits=model(x)
                pred=logits.argmax(dim=1)
                total_correct+=torch.eq(pred,label).float().sum().item()
                total+=x.size(0)
        print(epoch," test acc :",total_correct/total)


if __name__ == '__main__':
    main()