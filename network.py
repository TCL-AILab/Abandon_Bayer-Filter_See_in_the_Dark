import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

class CALayer(nn.Module):
    """
    Channel Attention Layer
    parameter: in_channel
    More detail refer to:
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Conv_LReLU_2(nn.Module):
    """
    Network component
    (Conv + LeakyReLU) × 2
    """
    def __init__(self, in_channel, out_channel):
        super(Conv_LReLU_2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.lrelu(self.Conv1(x))
        x = self.lrelu(self.Conv2(x))
        return x

class DBF_Module(nn.Module):
    """
    This is De-Bayer_Filter Module.
    Input: color RAW images of size (1 × H × W)
    Output: monochrome images of size (1 × H × W)
    """
    def __init__(self,):
        super(DBF_Module, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.DBF_conv1 = Conv_LReLU_2(1,8)
        self.DBF_conv2 = Conv_LReLU_2(8,16)
        self.DBF_conv3 = Conv_LReLU_2(16,32)
        self.DBF_conv4 = Conv_LReLU_2(32,64)
        self.DBF_conv5 = Conv_LReLU_2(64,128)
        self.DBF_upv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.DBF_conv6 = Conv_LReLU_2(128,64)
        self.DBF_upv6 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.DBF_conv7 = Conv_LReLU_2(64,32)
        self.DBF_upv7 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.DBF_conv8 = Conv_LReLU_2(32,16)
        self.DBF_upv8 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.DBF_conv9 = Conv_LReLU_2(16,8)
        self.DBF_out = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        DBF_conv1 = self.DBF_conv1(x)
        DBF_pool1 = self.pool1(DBF_conv1)

        DBF_conv2 = self.DBF_conv2(DBF_pool1)
        DBF_pool2 = self.pool1(DBF_conv2)

        DBF_conv3 = self.DBF_conv3(DBF_pool2)
        DBF_pool3 = self.pool1(DBF_conv3)

        DBF_conv4 = self.DBF_conv4(DBF_pool3)
        DBF_pool4 = self.pool1(DBF_conv4)

        DBF_conv5 = self.DBF_conv5(DBF_pool4)
        DBF_up5 = self.DBF_upv5(DBF_conv5)

        DBF_concat6 = torch.cat([DBF_up5, DBF_conv4], 1)
        DBF_conv6 = self.DBF_conv6(DBF_concat6)
        DBF_up6 = self.DBF_upv6(DBF_conv6)

        DBF_concat7 = torch.cat([DBF_up6, DBF_conv3], 1)
        DBF_conv7 = self.DBF_conv7(DBF_concat7)
        DBF_up7 = self.DBF_upv7(DBF_conv7)

        DBF_concat8 = torch.cat([DBF_up7, DBF_conv2], 1)
        DBF_conv8 = self.DBF_conv8(DBF_concat8)
        DBF_up8 = self.DBF_upv8(DBF_conv8)

        DBF_concat9 = torch.cat([DBF_up8, DBF_conv1], 1)
        DBF_conv9 = self.DBF_conv9(DBF_concat9)

        DBF_out = self.lrelu(self.DBF_out(DBF_conv9))

        return DBF_out

class DBLE_Module(nn.Module):
    """
    This is Dual Branch Low-light image Enhancement(DBLE) Module.
    Input: down-shuffled color and mono images of size (4 × H/2 × W/2)
    Output: shuffled RGB images of size (12 × H/2 × W/2)
    """
    def __init__(self,):
        super(DBLE_Module, self).__init__()
        self.up2 = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.mono_conv1 = Conv_LReLU_2(4, 32)
        self.mono_conv2 = Conv_LReLU_2(32, 64)
        self.mono_conv3 = Conv_LReLU_2(64, 128)
        self.mono_conv4 = Conv_LReLU_2(128, 256)
        self.mono_conv5 = Conv_LReLU_2(256, 512)
        self.mono_up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.color_conv1 = Conv_LReLU_2(4, 32)
        self.color_conv2 = Conv_LReLU_2(32, 64)
        self.color_conv3 = Conv_LReLU_2(64, 128)
        self.color_conv4 = Conv_LReLU_2(128, 256)
        self.color_conv5 = Conv_LReLU_2(256, 512)
        self.color_up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dual_conv6 = Conv_LReLU_2(1024, 256)
        self.dual_up6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dual_conv7 = Conv_LReLU_2(384, 128)
        self.dual_up7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dual_conv8 = Conv_LReLU_2(192, 64)
        self.dual_up8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dual_conv9 = Conv_LReLU_2(96, 32)
        self.DBLE_out = nn.Conv2d(32, 12, kernel_size=1, stride=1)

        self.channel_attention6 = CALayer(512 * 2)
        self.channel_attention7 = CALayer(384)
        self.channel_attention8 = CALayer(192)
        self.channel_attention9 = CALayer(96)

    def forward(self, color, mono):
        color_conv1 = self.color_conv1(color)
        color_pool1 = self.pool1(color_conv1)

        color_conv2 = self.color_conv2(color_pool1)
        color_pool2 = self.pool1(color_conv2)

        color_conv3 = self.color_conv3(color_pool2)
        color_pool3 = self.pool1(color_conv3)

        color_conv4 = self.color_conv4(color_pool3)
        color_pool4 = self.pool1(color_conv4)

        color_conv5 = self.color_conv5(color_pool4)
        color_up5 = self.color_up5(color_conv5)

        mono_conv1 = self.mono_conv1(mono)
        mono_pool1 = self.pool1(mono_conv1)

        mono_conv2 = self.mono_conv2(mono_pool1)
        mono_pool2 = self.pool1(mono_conv2)

        mono_conv3 = self.mono_conv3(mono_pool2)
        mono_pool3 = self.pool1(mono_conv3)

        mono_conv4 = self.mono_conv4(mono_pool3)
        mono_pool4 = self.pool1(mono_conv4)

        mono_conv5 = self.mono_conv5(mono_pool4)
        mono_up5 = self.mono_up5(mono_conv5)

        concat6 = torch.cat([color_up5, mono_up5, color_conv4, mono_conv4], 1)
        ca6 = self.channel_attention6(concat6)
        dual_conv6 = self.dual_conv6(ca6)
        dual_up6 = self.dual_up6(dual_conv6)
        
        concat7 = torch.cat([dual_up6, color_conv3, mono_conv3], 1)
        ca7 = self.channel_attention7(concat7)
        dual_conv7 = self.dual_conv7(ca7)
        dual_up7 = self.dual_up7(dual_conv7)
        
        concat8 = torch.cat([dual_up7, color_conv2, mono_conv2], 1)
        ca8 = self.channel_attention8(concat8)
        dual_conv8 = self.dual_conv8(ca8)
        dual_up8 = self.dual_up8(dual_conv8)
        
        concat9 = torch.cat([dual_up8, color_conv1, mono_conv1], 1)
        ca9 = self.channel_attention9(concat9)
        dual_conv9 = self.dual_conv9(ca9)

        DBLE_out = self.lrelu(self.DBLE_out(dual_conv9))

        return DBLE_out

def downshuffle(var, r):
    """
    Down Shuffle function.
    Input: variable of size (1 × H × W)
    Output: down-shuffled var of size (r^2 × H/r × W/r)
    """
    b, c, h, w = var.size()
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return var.contiguous().view(b, c, out_h, r, out_w, r)\
        .permute(0, 1, 3, 5, 2, 4).contiguous().view(b,out_channel,out_h,out_w).contiguous()

class our_Net(nn.Module):

    def __init__(self):
        super(our_Net, self).__init__()

        self.up2 = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.DBF = DBF_Module()
        self.DBLE = DBLE_Module()

    def forward(self, x):

        mono_out = self.DBF(x)

        DBLE_color = downshuffle(x,2)
        DBLE_mono = downshuffle(mono_out,2)

        DBLE_out = self.DBLE(DBLE_color, DBLE_mono)
        RGB_out = self.up2(DBLE_out)

        return (mono_out, RGB_out)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)



# # calculate GFLOPs and Parameters
# model = our_Net()
# ops, params = get_model_complexity_info(model, (1, 512, 512), as_strings=True,
#                                         print_per_layer_stat=True, verbose=True)
# print(ops, params)