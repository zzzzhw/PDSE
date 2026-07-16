#! /usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn

class SimpleEncClassifier(nn.Module):
    def __init__(self, enc_dims, mlp_dims, dropout=0.2, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        self.encoder_model = None
        self.mlp_model = None
        self.encoded = None
        self.mlp_out = None
        self.encoder_modules = []
        self.mlp_modules = []
        self.verbose = verbose
        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))

        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.encoder_model)
            print(self.mlp_model)
        return

    def update_mlp_head(self, dropout=0.2):
        self.mlp_out = None
        self.mlp_modules = []

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.encoder_model)
            print(self.mlp_model)
        return

    def forward(self, x):
        self.encoded = self.encoder_model(x)
        self.out = self.mlp_model(self.encoded)
        return self.encoded, self.encoded, self.out
    
    def predict_proba(self, x):
        _, _, mlp_out = self.forward(x)
        return mlp_out
    
    def predict(self, x):
        self.encoded = self.encoder_model(x)
        self.out = self.mlp_model(self.encoded)
        preds = self.out.max(1)[1]
        return preds
    
    def encode(self, x):
        self.encoded = self.encoder_model(x)
        return self.encoded

class Enc(nn.Module):
    def __init__(self, enc_dims, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.encoder_model = None
        self.encoded = None
        self.encoder_modules = []
        self.verbose = verbose
        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))
        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))

        if self.verbose:
            print(self.encoder_model)
        return

    def forward(self, x):
        self.encoded = self.encoder_model(x)
        return self.encoded
    
    def encode(self, x):
        self.encoded = self.encoder_model(x)
        return self.encoded

class CAE(nn.Module):
    def __init__(self, enc_dims, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.encoder_model = None
        self.decoder_model = None
        self.encoded = None
        self.decoded = None
        self.encoder_modules = []
        self.decoder_modules = []
        self.verbose = verbose
        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))
        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))
        self.encoder_model.apply(self.init_weights)

        # decoder
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            self.decoder_modules.append(nn.Linear(self.enc_dims[i + 1], self.enc_dims[i]))
            self.decoder_modules.append(nn.ReLU())
        # decoded output. no activation.
        self.decoder_modules.append(nn.Linear(self.enc_dims[1], self.enc_dims[0]))
        # decoder model
        self.decoder_model = nn.Sequential(*(self.decoder_modules))
        self.decoder_model.apply(self.init_weights)

        if self.verbose:
            print(self.encoder_model)
            print(self.decoder_model)
        return

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        return

    def forward(self, x):
        self.encoded = self.encoder_model(x)
        self.decoded = self.decoder_model(self.encoded)
        return self.encoded, self.decoded
    
    def encode(self, x):
        self.encoded = self.encoder_model(x)
        return self.encoded

class MLPClassifier(nn.Module):
    def __init__(self, mlp_dims, dropout=0.2, verbose=1):
        super().__init__()
        self.mlp_dims = mlp_dims
        self.mlp_model = None
        self.mlp_out = None
        self.mlp_modules = []
        self.verbose = verbose

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.mlp_model)
        return

    def forward(self, x):
        self.mlp_out = self.mlp_model(x)
        return self.mlp_out
    
    def predict_proba(self, x):
        mlp_out = self.forward(x)
        return mlp_out
    
    def predict(self, x):
        self.mlp_out = self.mlp_model(x)
        preds = self.mlp_out.max(1)[1]
        return preds
    
    def encode(self, x):
        self.encoded = self.mlp_model[:-2](x)
        return self.encoded

class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,  kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)

        return x
class ResModule(nn.Module):
    def __init__(self):
        super(ResModule, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x_ = self.conv1(x)
        x_ = self.bn1(x_)
        x_ = F.relu(x_)
        x_ = self.conv2(x_)
        x_ = self.bn2(x_)
        x = F.relu(x_ + x)

        return x
class AdaptiveAvgPool2dModule(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, self.output_size)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
## ResNet神经网络
class ResClassifier(nn.Module):
    def __init__(self,outputdim):
        super(ResClassifier, self).__init__()
        self.mlp_out = None
        self.mlp_model = None
        self.cnn = CNNModule()
        self.res1 = ResModule()
        self.res2 = ResModule()
        self.res3 = ResModule()
        self.res4 = ResModule()
        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, outputdim)

        # 将模型组件添加到列表中
        self.mlp_modules = [
            self.cnn,
            self.res1,
            self.res2,
            self.res3,
            self.res4,
            AdaptiveAvgPool2dModule((5, 5)),
            Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.Softmax(dim=1)
        ]

        # 创建顺序模型
        self.mlp_model = nn.Sequential(*self.mlp_modules)

    def forward(self, x):
        self.mlp_out = self.mlp_model(x)
        return self.mlp_out

    def predict_proba(self, x):
        mlp_out = self.forward(x)
        return  mlp_out
    def predict(self, x):
        self.mlp_out = self.mlp_model(x)
        preds = self.mlp_out.max(1)[1]
        return preds

    def encode(self, x):
        self.encoded = self.mlp_model[:-2](x)
        return self.encoded
