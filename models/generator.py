import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,downsample=True,activation='relu',use_dropout=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,4,2,1,padding_mode="reflect") if downsample else \
            nn.ConvTranspose2d(in_channels,out_channels,4,2,1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.act =nn.ReLU() if activation=="relu" else nn.LeakyReLU(0.2)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x=self.conv(x)
        x=self.batchnorm(x)
        x=self.act(x)
        return self.dropout(x) if self.use_dropout else x




class Generator(nn.Module):
    def __init__(self,in_channels,features =64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,features,4,2,1,padding_mode="reflect")
        self.act = nn.LeakyReLU(0.2)


        self.down1 = CNNBlock(features,features*2,downsample=True,activation="leaky")
        self.down2 = CNNBlock(features*2, features * 4, downsample=True, activation="leaky")
        self.down3 = CNNBlock(features*4, features * 8, downsample=True, activation="leaky")
        self.down4 = CNNBlock(features*8, features * 8, downsample=True, activation="leaky")
        self.down5 = CNNBlock(features * 8, features * 8, downsample=True, activation="leaky")
        self.down6 = CNNBlock(features * 8, features * 8, downsample=True, activation="leaky")
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1),nn.ReLU()
        )
        self.up1 = CNNBlock(features*8,features*8,downsample=False,activation="relu",use_dropout=True)
        self.up2 = CNNBlock(features * 8*2, features * 8, downsample=False, activation="relu", use_dropout=True)
        self.up3 = CNNBlock(features * 8*2, features * 8, downsample=False, activation="relu", use_dropout=True)
        self.up4 = CNNBlock(features * 8*2, features * 8, downsample=False, activation="relu", use_dropout=True)
        self.up5 = CNNBlock(features * 8*2, features*4, downsample=False, activation="relu", use_dropout=True)
        self.up6 = CNNBlock(features * 4* 2, features*2, downsample=False, activation="relu", use_dropout=True)
        self.up7 = CNNBlock(features * 2 * 2, features, downsample=False, activation="relu", use_dropout=True)
        self.up = nn.ConvTranspose2d(features*2,in_channels,kernel_size=4,stride=2,padding=1)
        self.tan = nn.Tanh()

    def forward(self,x):
        d= self.conv(x)
        d= self.act(d)
        d1= self.down1(d)
        d2=self.down2(d1)
        d3=self.down3(d2)
        d4= self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        # print(d6.shape)
        bottlenect = self.bottleneck(d6)
        up1 = self.up1(bottlenect)
        up2 = self.up2(torch.cat([up1,d6],1))
        up3 = self.up3(torch.cat([up2,d5],1))
        up4 = self.up4(torch.cat([up3,d4],1))
        up5 = self.up5(torch.cat([up4,d3],1))
        up6 = self.up6(torch.cat([up5,d2],1))
        up7 = self.up7(torch.cat([up6,d1],1))
        return self.up(torch.cat([up7,d],1))


generator = Generator(in_channels=3)