import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=stride,padding_mode="reflect")
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x = self.conv(x)
        x= self.batchNorm(x)
        x=self.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,in_channels = 3,features =[64,128,256,512]):
        super().__init__()
        self.conv =nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        self.lRelu = nn.LeakyReLU(0.2)
        layers = []
        in_channels=features[0]

        for feature in features[1:]:
            layers.append(
                block(in_channels,feature,stride=1 if feature == features[-1] else 2)
            )
            in_channels=feature

        layers.append(
            nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding =1 ,padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

        ## y fake or y real
    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        output = self.conv(input)
        output = self.lRelu(output)
        output = self.model(output)
        return output

discriminator = Discriminator(in_channels=3)