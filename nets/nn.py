import math

import torch


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, d, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.add_m = s != 1 or in_ch != out_ch

        self.conv1 = Conv(in_ch, out_ch, torch.nn.ReLU(), k=3, s=s, p=1)
        self.conv2 = Conv(out_ch, out_ch, torch.nn.Identity(), k=3, s=1, p=1)

        if self.add_m:
            self.conv3 = Conv(in_ch, out_ch, torch.nn.Identity(), s=s)

    def zero_init(self):
        torch.nn.init.zeros_(self.conv2.norm.weight)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.add_m:
            x = self.conv3(x)

        return self.relu(x + y)


class ResNet(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU(), k=7, s=2, p=3))
        # p2/4
        self.p2.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.p2.append(Residual(filters[1], filters[1], s=1))
        self.p2.append(Residual(filters[1], filters[1], s=1))
        # p3/8
        self.p3.append(Residual(filters[1], filters[2], s=2))
        self.p3.append(Residual(filters[2], filters[2], s=1))
        # p4/16
        self.p4.append(Residual(filters[2], filters[3], s=2))
        self.p4.append(Residual(filters[3], filters[3], s=1))
        # p5/32
        self.p5.append(Residual(filters[3], filters[4], s=2))
        self.p5.append(Residual(filters[4], filters[4], s=1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p1, p2, p3, p4, p5


class SCPConv(torch.nn.Module):
    def __init__(self, ch, dilation, width=24):
        super().__init__()
        self.width = int(math.floor(ch * (width / 64.0)))
        self.conv1 = Conv(ch, 4 * self.width, torch.nn.ReLU())

        self.conv2 = Conv(self.width, self.width, torch.nn.ReLU(), k=3, p=dilation[0], d=dilation[0])
        self.conv3 = Conv(self.width, self.width, torch.nn.ReLU(), k=3, p=dilation[1], d=dilation[1])
        self.conv4 = Conv(self.width, self.width, torch.nn.ReLU(), k=3, p=dilation[2], d=dilation[2])
        self.conv5 = Conv(self.width, self.width, torch.nn.ReLU(), k=3, p=dilation[3], d=dilation[3])
        self.conv6 = Conv(4 * self.width, ch, torch.nn.Identity())

    def zero_init(self):
        torch.nn.init.zeros_(self.conv6.norm.weight)

    def forward(self, x):
        y = self.conv1(x).chunk(4, 1)

        y1 = self.conv2(y[0])
        y2 = self.conv3(y[1] + y1)
        y3 = self.conv4(y[2] + y2)
        y4 = self.conv5(y[3] + y3)
        return self.conv6(torch.cat(tensors=(y1, y2, y3, y4), dim=1)) + x


class EDB(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        # p6/64
        self.p6 = torch.nn.Sequential(Residual(filters[4], filters[5], s=2),
                                      Residual(filters[5], filters[5], s=1))
        # p7/128
        self.p7 = torch.nn.Sequential(Residual(filters[5], filters[6], s=2),
                                      Residual(filters[6], filters[6], s=1))
        self.conv1 = Conv(filters[5], filters[5] // 2, torch.nn.ReLU())
        self.conv2 = Conv(filters[6], filters[6] // 2, torch.nn.ReLU())
        self.conv3 = Conv(filters[6], filters[6], torch.nn.ReLU())
        self.conv4 = torch.nn.ModuleList([SCPConv(filters[6], width=32, dilation=[1, 1, 1, 1]),
                                          SCPConv(filters[5], width=32, dilation=[1, 1, 1, 1])])

        self.sigmoid = torch.nn.Sigmoid()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(p6)

        y = self.avg_pool(p7)
        y = self.sigmoid(y)

        p7 = p7 * y
        p6 = p6 * y

        p7 = self.conv3(p7)
        p7 = self.conv4[0](p7)

        p6 = self.conv1(p6)
        p7 = self.conv2(p7)
        up = torch.nn.functional.interpolate(p7,
                                             size=p6.shape[2:],
                                             mode='bilinear',
                                             align_corners=False)
        p6 = torch.cat(tensors=(up, p6), dim=1)
        return self.conv4[1](p6)


class Neck(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.edb = EDB(filters)
        dilation = [[1, 2, 3, 4]] * 2 + [[1, 2, 4, 8]] * 3

        self.conv1 = torch.nn.ModuleList([Conv(filters[4], filters[4] // 2, torch.nn.ReLU()),
                                          Conv(filters[3], filters[3] // 2, torch.nn.ReLU()),
                                          Conv(filters[2], filters[2] // 2, torch.nn.ReLU()),
                                          Conv(filters[1], filters[1] // 2, torch.nn.ReLU()),
                                          Conv(filters[1], filters[1] // 2, torch.nn.ReLU())])

        self.conv2 = torch.nn.ModuleList([Conv(filters[5], filters[4] // 2, torch.nn.ReLU()),
                                          Conv(filters[4], filters[3] // 2, torch.nn.ReLU()),
                                          Conv(filters[3], filters[2] // 2, torch.nn.ReLU()),
                                          Conv(filters[2], filters[1] // 2, torch.nn.ReLU()),
                                          Conv(filters[1], filters[1] // 2, torch.nn.ReLU())])

        self.conv3 = torch.nn.ModuleList([torch.nn.Sequential(SCPConv(filters[4], dilation[0]),
                                                              SCPConv(filters[4], dilation[0])),
                                          torch.nn.Sequential(SCPConv(filters[3], dilation[1]),
                                                              SCPConv(filters[3], dilation[1])),
                                          torch.nn.Sequential(SCPConv(filters[2], dilation[2]),
                                                              SCPConv(filters[2], dilation[2])),
                                          torch.nn.Sequential(SCPConv(filters[1], dilation[3]),
                                                              SCPConv(filters[1], dilation[3])),
                                          torch.nn.Sequential(SCPConv(filters[1], dilation[4]),
                                                              SCPConv(filters[1], dilation[4]))])

    def forward(self, x):
        p6 = self.edb(x[4])
        p5 = self.conv1[0](x[4])
        p4 = self.conv1[1](x[3])
        p3 = self.conv1[2](x[2])
        p2 = self.conv1[3](x[1])
        p1 = self.conv1[4](x[0])

        up = torch.nn.functional.interpolate(self.conv2[0](p6),
                                             size=p5.shape[2:],
                                             mode='bilinear',
                                             align_corners=False)
        p5 = torch.cat(tensors=(up, p5), dim=1)
        p5 = self.conv3[0](p5)
        up = torch.nn.functional.interpolate(self.conv2[1](p5),
                                             size=p4.shape[2:],
                                             mode='bilinear',
                                             align_corners=False)
        p4 = torch.cat(tensors=(up, p4), dim=1)
        p4 = self.conv3[1](p4)
        up = torch.nn.functional.interpolate(self.conv2[2](p4),
                                             size=p3.shape[2:],
                                             mode='bilinear',
                                             align_corners=False)
        p3 = torch.cat(tensors=(up, p3), dim=1)
        p3 = self.conv3[2](p3)
        up = torch.nn.functional.interpolate(self.conv2[3](p3),
                                             size=p2.shape[2:],
                                             mode='bilinear',
                                             align_corners=False)
        p2 = torch.cat(tensors=(up, p2), dim=1)
        p2 = self.conv3[3](p2)
        up = torch.nn.functional.interpolate(self.conv2[4](p2),
                                             size=p1.shape[2:],
                                             mode='bilinear',
                                             align_corners=False)
        p1 = torch.cat(tensors=(up, p1), dim=1)
        p1 = self.conv3[4](p1)
        return p1, p2, p3, p4, p5


class Head(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.fc1 = torch.nn.Conv2d(filters[1], out_channels=1, kernel_size=1)
        self.fc2 = torch.nn.Conv2d(filters[1], out_channels=1, kernel_size=1)
        self.fc3 = torch.nn.Conv2d(filters[2], out_channels=1, kernel_size=1)
        self.fc4 = torch.nn.Conv2d(filters[3], out_channels=1, kernel_size=1)
        self.fc5 = torch.nn.Conv2d(filters[4], out_channels=1, kernel_size=1)

    def forward(self, x):
        p1 = self.fc1(x[0])
        p2 = self.fc2(x[1])
        p3 = self.fc3(x[2])
        p4 = self.fc4(x[3])
        p5 = self.fc5(x[4])
        return p1, p2, p3, p4, p5


class EDNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        filters = [3, 64, 128, 256, 512, 256, 256]
        self.backbone = ResNet(filters)
        self.neck = Neck(filters)
        self.head = Head(filters)

        # initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        for m in self.modules():
            if hasattr(m, 'zero_init'):
                m.zero_init()

    def forward(self, x):
        y = self.backbone(x)
        y = self.neck(y)
        y = self.head(y)

        p1 = torch.nn.functional.interpolate(y[0],
                                             x.shape[2:],
                                             mode='bilinear',
                                             align_corners=False)
        if self.training:
            p2 = torch.nn.functional.interpolate(y[1],
                                                 x.shape[2:],
                                                 mode='bilinear',
                                                 align_corners=False)
            p3 = torch.nn.functional.interpolate(y[2],
                                                 x.shape[2:],
                                                 mode='bilinear',
                                                 align_corners=False)
            p4 = torch.nn.functional.interpolate(y[3],
                                                 x.shape[2:],
                                                 mode='bilinear',
                                                 align_corners=False)
            p5 = torch.nn.functional.interpolate(y[4],
                                                 x.shape[2:],
                                                 mode='bilinear',
                                                 align_corners=False)
            return torch.cat(tensors=(p1, p2, p3, p4, p5), dim=1)
        else:
            return p1
