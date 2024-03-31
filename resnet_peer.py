import torch
import torch.nn as nn





def conv3x3x3(in_channels, out_channels, stride=1, padding=1):
    '''3d convolution with padding'''
    return nn.Conv3d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=False)


def conv1x1x1(in_channels, out_channels, stride=1):
    '''1x1x1 convolution'''
    return nn.Conv3d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, block, layers, n_classes = 1):
        self.inplanes = 64
        super(Resnet, self).__init__()
        # self.in_channels = 3

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5,4,4), stride=(1, 1, 1), padding=(1, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool3d((2,2,2))
        self.fc1 = nn.Linear(512*block.expansion,2048)
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512,n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride=stride
                ),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)
        x = self.dropout2(x)
        x = self.layer4(x)
        x = self.dropout2(x)
        x = self.avgpool(x)

        x = x.view(x.size(0),-1)
        #x = torch.cat([x, centroid], dim=1)
        #x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.fc3(x)


        return x
def resnet5():
    return Resnet(BasicBlock,[1,1,0,0])

def resnet10():
    return Resnet(BasicBlock, [1, 1, 1, 1])
def resnet18():
    return Resnet(BasicBlock, [2, 2, 2, 2])

def convert_to_pix(label):
    monitor_width = 1680
    monitor_height = 1050
    w_dpi = 123
    h_dpi = 127
    coe = 25.4
    D = 1350

    x_label = label[:,0]
    y_label = label[:,1]

    half_x_H = x_label*monitor_width / w_dpi * coe /2
    half_y_H = y_label*monitor_height/ h_dpi * coe /2

    x_visual_angle = torch.atan(half_x_H/D) * 180 / 3.14
    y_visual_angle = torch.atan(half_y_H/D) * 180 / 3.14

    x_visual_angle = x_visual_angle.unsqueeze(1)




    # x_label = x_label.unsqueeze(1)
    return x_visual_angle

def trainer(net,train_loader,optimizer, epoch, loss_func):

    net.train()
    sum_loss = 0.0
    for id, (img, labels,centroid,bbox) in enumerate(train_loader):
        #centroid = centroid.view(centroid.size(0),6).cuda()/84.0
        #bbox = bbox.view(bbox.size(0),12).cuda()/84.0
        img = img.unsqueeze(1).cuda()
        #result = net(img,centroid,bbox)
        result = net(img)
        labels = convert_to_pix(labels)
        labels = labels.cuda()
        loss = loss_func(labels,result)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        if (epoch+1)/5 == 0:
            print("train:"+str(epoch+1)+','+str(id)+','+str(loss.item()))
    sum_loss = sum_loss /(id+1)
    print("################################train_loss=%.3f" % sum_loss)
    return sum_loss


def valer(net,val_loader,loss_func):
    net.eval()
    with torch.no_grad():
        sum_loss = 0.0
        for id, (img, labels, centroid, bbox) in enumerate(val_loader):
            #centroid = centroid.view(centroid.size(0), 6).cuda()/84.0
            #bbox = bbox.view(bbox.size(0), 12).cuda()/84.0
            img = img.unsqueeze(1)
            img = img.cuda()
            #out = net(img,centroid,bbox)
            out = net(img)
            labels = convert_to_pix(labels)
            labels = labels.cuda()
            loss = loss_func(out,labels)
            sum_loss += loss.item()
        sum_loss = sum_loss/len(val_loader)
        print("################################evl_loss=%.3f"%sum_loss)
    return sum_loss
