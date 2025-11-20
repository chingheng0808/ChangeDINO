'''
@misc{woo2018cbamconvolutionalblockattention,
      title={CBAM: Convolutional Block Attention Module}, 
      author={Sanghyun Woo and Jongchan Park and Joon-Young Lee and In So Kweon},
      year={2018},
      eprint={1807.06521},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1807.06521}, 
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            bias=self.bias,
        )

    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        struct = self.conv(torch.cat((max, avg), dim=1))
        x = F.sigmoid(struct) * x
        return x


class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.channels,
                out_features=self.channels // self.r,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.channels // self.r,
                out_features=self.channels,
                bias=True,
            ),
        )

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        max = self.linear(max.view(b, c)).view(b, c, 1, 1)
        avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
        x = F.sigmoid(max + avg) * x
        return x


class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x
