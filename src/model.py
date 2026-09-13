from torch import nn

class HouseRentPrice(nn.Module):
    def __init__(self,input_features):
        super().__init__()
        self.layer1=nn.Linear(input_features,out_features=128)
        self.layer2=nn.Linear(in_features=128,out_features=64)
        self.layer3=nn.Linear(in_features=64,out_features=32)
        self.layer4=nn.Linear(in_features=32,out_features=1)

        self.Relu=nn.ReLU()
    def forward(self,x):
        return self.layer4(self.Relu(self.layer3(self.Relu(self.layer2(self.Relu(self.layer1(x)))))))