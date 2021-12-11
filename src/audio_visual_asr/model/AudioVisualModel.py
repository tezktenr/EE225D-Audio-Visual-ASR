"""
Filename: AudioVisualModel.py
Description: This is a file that contains all the neural network models for audio and video training
"""

# Python Standard Libraries
import math

# Third Party Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable

# Project Module


# Source Code
#######################################################################
class GRU(nn.Module):
    """
    multi-layer gated recurrent unit (GRU) RNN
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True, use_gpu=False):
        super(GRU, self).__init__()
        # GRU model details
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

        # GPU settings
        self.setGPU(use_gpu)
    
    def setGPU(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            super(GRU, self).cuda()

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        if self.use_gpu:
            h0 = h0.cuda()
        out, _ = self.gru(x, h0)
        # if self.every_frame:
        #     out = self.fc(out)  # predicitions based on every time step
        # else:
        #     out = self.fc(out[:, -1, :])  # predictions based on the last time step
        return out

#######################################################################
class AudioBasicBlock(nn.Module):
    """
    A basic block used in RESNET for audio data
    """

    expansion = 1

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv1d(in_planes, out_planes,
                         kernel_size=3, stride=stride, padding=1, bias=False)

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_gpu=False):
        super(AudioBasicBlock, self).__init__()
        # AudioBasicBlock model details
        self.conv1 = AudioBasicBlock.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = AudioBasicBlock.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

        # GPU settings
        self.setGPU(use_gpu)
    
    def setGPU(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            super(AudioBasicBlock, self).cuda()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


#######################################################################
class AudioResNet(nn.Module):
    """
    RESNET for audio data
    """

    def __init__(self, block, layers, num_classes=1000, use_gpu=False):
        super(AudioResNet, self).__init__()
        # AudioResNet model details
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=21, padding=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # GPU settings
        self.setGPU(use_gpu)

    def setGPU(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            super(AudioResNet, self).cuda()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, x.size(2))
        x = self.fc(x)
        return x



#######################################################################
class AudioRecognition(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, every_frame=True, use_gpu=False):
        super(AudioRecognition, self).__init__()
        # AudioRecognition model details
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.every_frame = every_frame
        self.nLayers = 2

        # frontend1D
        self.fronted1D = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(True)
        )

        # resnet
        self.resnet18 = AudioResNet(AudioBasicBlock, [2, 2, 2, 2], num_classes=self.inputDim, use_gpu=self.use_gpu)

        # backend_conv
        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(self.inputDim, 2*self.inputDim, 5, 2, 0, bias=False),
            nn.BatchNorm1d(2*self.inputDim),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(2*self.inputDim, 4*self.inputDim, 5, 2, 0, bias=False),
            nn.BatchNorm1d(4*self.inputDim),
            nn.ReLU(True),
        )

        self.backend_conv2 = nn.Sequential(
            nn.Linear(4*self.inputDim, self.inputDim),
            nn.BatchNorm1d(self.inputDim),
            nn.ReLU(True),
            nn.Linear(self.inputDim, self.nClasses)
        )

        # backend_gru
        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.every_frame, use_gpu=self.use_gpu)

        # initialize
        self._initialize_weights()

        # GPU settings
        self.setGPU(use_gpu)
    
    def setGPU(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            super(AudioRecognition, self).cuda()
            self.resnet18.setGPU(self.use_gpu)
            self.gru.setGPU(self)

    def forward(self, x):
        x = x.view(-1, 1, x.size(1))
        x = self.fronted1D(x)
        x = x.contiguous()
        x = self.resnet18(x)
        if self.mode == 'temporalConv':
            x = x.view(-1, self.frameLen, self.inputDim)
            x = x.transpose(1, 2)
            x = self.backend_conv1(x)
            x = torch.mean(x, 2)
            x = self.backend_conv2(x)
        elif self.mode == 'backendGRU' or self.mode == 'finetuneGRU':
            x = x.view(-1, self.frameLen, self.inputDim)
            x = self.gru(x)
        else:
            raise Exception('No model is selected')
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


#######################################################################
class VideoBasicBlock(nn.Module):
    """
    A basic block used in RESNET for video data
    """

    expansion = 1

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes,
                         kernel_size=3, stride=stride, padding=1, bias=False)

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_gpu=False):
        super(VideoBasicBlock, self).__init__()
        # VideoBasicBlock model details
        self.conv1 = VideoBasicBlock.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = VideoBasicBlock.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # GPU settings
        self.setGPU(use_gpu)
    
    def setGPU(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            super(VideoBasicBlock, self).cuda()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




#######################################################################
class VideoResNet(nn.Module):
    """
    RESNET for video data
    """

    def __init__(self, block, layers, num_classes=1000, use_gpu=False):
        super(VideoResNet, self).__init__()
        # VideoResNet model details
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bnfc = nn.BatchNorm1d(num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # GPU settings
        self.setGPU(use_gpu)
    
    def setGPU(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            super(VideoResNet, self).cuda()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bnfc(x)
        return x



#######################################################################
class LipReading(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, every_frame=True, use_gpu=False):
        super(LipReading, self).__init__()
        # LipReading model details
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.every_frame = every_frame
        self.nLayers = 2

        # frontend3D
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # resnet
        self.resnet34 = VideoResNet(VideoBasicBlock, [3, 4, 6, 3], num_classes=self.inputDim)

        # backend_conv
        self.backend_conv1 = nn.Sequential(
                nn.Conv1d(self.inputDim, 2*self.inputDim, 5, 2, 0, bias=False),
                nn.BatchNorm1d(2*self.inputDim),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(2*self.inputDim, 4*self.inputDim, 5, 2, 0, bias=False),
                nn.BatchNorm1d(4*self.inputDim),
                nn.ReLU(True),
        )
        self.backend_conv2 = nn.Sequential(
                nn.Linear(4*self.inputDim, self.inputDim),
                nn.BatchNorm1d(self.inputDim),
                nn.ReLU(True),
                nn.Linear(self.inputDim, self.nClasses)
        )

        # backend_gru
        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.every_frame)

        # initialize
        self._initialize_weights()

        # GPU settings
        self.setGPU(use_gpu)

    def setGPU(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            super(LipReading, self).cuda()
            self.resnet34.setGPU(self.use_gpu)
            self.gru.setGPU(self.use_gpu)
        

    def forward(self, x):
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet34(x)
        if self.mode == 'temporalConv':
            x = x.view(-1, self.frameLen, self.inputDim)
            x = x.transpose(1, 2)
            x = self.backend_conv1(x)
            x = torch.mean(x, 2)
            x = self.backend_conv2(x)
        elif self.mode == 'backendGRU' or self.mode == 'finetuneGRU':
            x = x.view(-1, self.frameLen, self.inputDim)
            x = self.gru(x)
        else:
            raise Exception('No model is selected')
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


#######################################################################
class ConcatGRU(nn.Module):
    """
    multi-layer gated recurrent unit (GRU) RNN for concat model
    """

    def __init__(self, inputDim=2048, hiddenDim=512, nLayers=2, nClasses=500, every_frame=True, use_gpu=False):
        super(ConcatGRU, self).__init__()
        # ConcatGRU model details
        self.hidden_size = hiddenDim
        self.num_layers = nLayers
        self.every_frame = every_frame
        self.gru = nn.GRU(inputDim, hiddenDim, nLayers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hiddenDim*2, nClasses)

        # GPU settings
        self.setGPU(use_gpu)

    def setGPU(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            super(ConcatGRU, self).cuda()

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        if self.use_gpu:
            h0 = h0.cuda()
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        if self.every_frame:
            out = self.fc(out)  # predictions based on every time step
        else:
            out = self.fc(out[:, -1, :])  # predictions based on last time-step
        return out


# For Testing Purposes
if __name__ == "__main__":
    pass