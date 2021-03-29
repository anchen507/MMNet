import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import PixelUnshuffle
import numpy as np

class Net(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, nf=64):
        super(Net, self).__init__()
        self.in_channel = in_channel
        self.spatiotemporal_feature = spatiotemporal_denoising(filter_in=64,filter_out=64,groups=1)
        self.spatial_feature = spatial_denoising(filter_in=64,filter_out=64,groups=1)
        self.esimator = merging_module(filter_in=64,filter_out=64,groups=1)

    def forward(self, x):
        N,C,L,H,W = x.size()
        residual=x
        out1 = self.spatiotemporal_feature(x,[N,C,L,H,W])
        out2 = self.spatial_feature(x,[N,C,L,H,W])
        out3 = torch.cat((out1,out2),1)
        out = self.esimator (out3)
        out = torch.add(out, residual)
        return out



class down_sample(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(down_sample, self).__init__()
        self.conv = nn.Conv3d(filter_in*4, filter_out , (1,1,1), 1, 0, groups=groups, bias=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input, shape):
        N,C,L,H,W=shape
        out = input.permute(0,2,1,3,4).reshape(N*L,-1,H,W)
        out = PixelUnshuffle.pixel_unshuffle(out,2)
        out = out.reshape(N,L,-1,H//2,W//2).permute(0,2,1,3,4)
        out = self.lrule(self.conv(out))

        return out


class up_sample(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(up_sample, self).__init__()
        self.conv = nn.Conv3d(filter_in, filter_out*4 , (1,1,1), 1, 0, groups=groups, bias=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input, shape):
        N,C,L,H,W=shape
        out = self.lrule(self.conv(input))
        out = out.permute(0,2,1,3,4).reshape(N*L,-1,H,W)
        out = F.pixel_shuffle(out,2)
        out = out.reshape(N,L,-1,H*2,W*2).permute(0,2,1,3,4)
        return out


class Seq_conv(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,concat_filter=64, groups=1):
        super(Seq_conv, self).__init__()
        self.conv1 = nn.Conv3d(concat_filter, filter_out , (1,1,1), 1, 0, groups=groups, bias=True)
        self.conv2 = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)
        self.conv3 = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)

        self.BN2 = nn.BatchNorm3d(filter_in, affine=True)
        self.BN3 = nn.BatchNorm3d(filter_in, affine=True)

        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input):
        residual=self.conv1(input)
        out=self.lrule(self.BN2(self.conv2(residual)))
        out=self.lrule(self.BN3(self.conv3(out)))
        return out+residual


class Seq_conv_tail(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(Seq_conv_tail, self).__init__()
        self.conv = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)
        self.BN = nn.BatchNorm3d(filter_in, affine=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input):
        out = self.lrule(self.BN(self.conv(input)))
        return out

class Seq_conv_ST(nn.Module):
     def __init__(self, filter_in=64,filter_out=64, concat_filter=64, groups=1):
        super(Seq_conv_ST, self).__init__()
        self.conv1 = nn.Conv3d(concat_filter, filter_out , (1,1,1), 1, 0, groups=groups, bias=True)
        self.conv2 = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)
        self.conv2_T = nn.Conv3d(filter_in, filter_out , (3,1,1), 1, (1,0,0), groups=groups, bias=False)
        self.conv3 = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)
        self.conv3_T = nn.Conv3d(filter_in, filter_out , (3,3,3), 1, (1,1,1), groups=groups, bias=False)

        self.BN2 = nn.BatchNorm3d(filter_in, affine=True)
        self.BN3 = nn.BatchNorm3d(filter_in, affine=True)

        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input):
        residual=self.conv1(input)
        out=self.lrule(self.conv2(residual))
        out=self.lrule(self.BN2(self.conv2_T(out)))
        out=self.lrule(self.conv3(out))
        out=self.lrule(self.BN3(self.conv3_T(out)))
        return out+residual


class Seq_conv_tail_ST(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(Seq_conv_tail_ST, self).__init__()
        self.conv = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)
        self.conv_T = nn.Conv3d(filter_in, filter_out , (3,3,3), 1, (1,1,1), groups=groups, bias=False)
        self.BN = nn.BatchNorm3d(filter_in, affine=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input):
        out = self.lrule(self.conv(input))
        out = self.lrule(self.BN(self.conv_T(out)))
        return out


class spatiotemporal_denoising(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(spatiotemporal_denoising, self).__init__()
        self.input_data = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
		
        self.seqconv1=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling1=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv2=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling2=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv3=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling3=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv4=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.up_sampling1=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv5=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.up_sampling2=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv6=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.up_sampling3=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv7=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.seqconv8=Seq_conv_tail_ST(filter_in=64,filter_out=64, groups=groups)

     def forward(self, input, shape):
        
        N,C,L,H,W=shape

        input_ = self.input_data(input)
		
		
        conc1 = self.seqconv1(input_)

        out = self.down_sampling1(conc1, [N,C,L,H,W])

        conc2 = self.seqconv2(out)

        out = self.down_sampling2(conc2, [N,C,L,H//2,W//2])

        conc3 = self.seqconv3(out)

        out = self.down_sampling3(conc3, [N,C,L,H//4,W//4])

        out = self.seqconv4(out)

        out = self.up_sampling1(out, [N,C,L,H//8,W//8])

        out = torch.cat((out,conc3),1)

        out = self.seqconv5(out)

        out = self.up_sampling2(out, [N,C,L,H//4,W//4])

        out = torch.cat((out,conc2),1)

        out = self.seqconv6(out)

        out = self.up_sampling3(out, [N,C,L,H//2,W//2])

        out = torch.cat((out,conc1),1)

        out = self.seqconv7(out)

        out = self.seqconv8(out)

        return out

class spatial_denoising(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(spatial_denoising, self).__init__()
        self.input_data = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
		
        self.seqconv1=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling1=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv2=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling2=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv3=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling3=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv4=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.up_sampling1=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv5=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.up_sampling2=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv6=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.up_sampling3=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv7=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.seqconv8=Seq_conv_tail(filter_in=64,filter_out=64, groups=groups)

     def forward(self, input, shape):
        
        N,C,L,H,W=shape

        input_ = self.input_data(input)
		
        conc1 = self.seqconv1(input_)

        out = self.down_sampling1(conc1, [N,C,L,H,W])

        conc2 = self.seqconv2(out)

        out = self.down_sampling2(conc2, [N,C,L,H//2,W//2])

        conc3 = self.seqconv3(out)

        out = self.down_sampling3(conc3, [N,C,L,H//4,W//4])

        out = self.seqconv4(out)

        out = self.up_sampling1(out, [N,C,L,H//8,W//8])

        out = torch.cat((out,conc3),1)

        out = self.seqconv5(out)

        out = self.up_sampling2(out, [N,C,L,H//4,W//4])

        out = torch.cat((out,conc2),1)

        out = self.seqconv6(out)

        out = self.up_sampling3(out, [N,C,L,H//2,W//2])

        out = torch.cat((out,conc1),1)

        out = self.seqconv7(out)

        out = self.seqconv8(out)

        return out


class merging_module(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,inchannel=3,groups=1):
        super(merging_module, self).__init__()

        self.Seq_conv1 = Seq_conv(filter_in=filter_in,filter_out=filter_out, concat_filter=filter_in*2, groups=groups)

        self.conv2 = nn.Conv3d(filter_out, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)

        self.Seq_conv3 = Seq_conv(filter_in=filter_out,filter_out=filter_out, concat_filter=filter_out, groups=groups)

        self.conv4 = nn.Conv3d(filter_out, 3 , (1,3,3), 1, (0,1,1), groups=groups, bias=False)

        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.BN = nn.BatchNorm3d(filter_out, affine=True)

     def forward(self, input):

        out = self.Seq_conv1(input)

        out = self.lrule(self.BN(self.conv2(out)))

        out = self.Seq_conv3(out)

        out=  self.conv4(out)

        return out




if __name__ == "__main__":
    net = Net().cuda()
    from thop import profile
    input = torch.randn(1, 1, 7, 224, 448).cuda()
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))


