import numpy as np
import torch
import torch.nn as nn

def wrap_flow_and_cal_diff(x,gt,flow):
     
    N,C,L,H,W = x.size()
    [frame1,frame2,frame3,frame4,frame5,frame6,frame7]=torch.split(x,1,dim=2)

    [gt_frame1,gt_frame2,gt_frame3,gt_frame4,gt_frame5,gt_frame6,gt_frame7]=torch.split(gt,1,dim=2)

    [flow1,flow2,flow3,flow4,flow5,flow6]=torch.split(flow,1,dim=2)

    wrap_frame2,mask2=warp(frame1,flow1)
    wrap_frame3,mask3=warp(frame2,flow2)
    wrap_frame4,mask4=warp(frame3,flow3)
    wrap_frame5,mask5=warp(frame4,flow4)
    wrap_frame6,mask6=warp(frame5,flow5)
    wrap_frame7,mask7=warp(frame6,flow6)

    gt_wrap_frame2,gt_mask2=warp(gt_frame1,flow1)
    gt_wrap_frame3,gt_mask3=warp(gt_frame2,flow2)
    gt_wrap_frame4,gt_mask4=warp(gt_frame3,flow3)
    gt_wrap_frame5,gt_mask5=warp(gt_frame4,flow4)
    gt_wrap_frame6,gt_mask6=warp(gt_frame5,flow5)
    gt_wrap_frame7,gt_mask7=warp(gt_frame6,flow6)

    diff_frame12=(frame2-wrap_frame2)*mask2
    diff_frame23=(frame3-wrap_frame3)*mask3
    diff_frame34=(frame4-wrap_frame4)*mask4
    diff_frame45=(frame5-wrap_frame5)*mask5
    diff_frame56=(frame6-wrap_frame6)*mask6
    diff_frame67=(frame7-wrap_frame7)*mask7

    gt_diff_frame12=(gt_frame2-gt_wrap_frame2)*gt_mask2
    gt_diff_frame23=(gt_frame3-gt_wrap_frame3)*gt_mask3
    gt_diff_frame34=(gt_frame4-gt_wrap_frame4)*gt_mask4
    gt_diff_frame45=(gt_frame5-gt_wrap_frame5)*gt_mask5
    gt_diff_frame56=(gt_frame6-gt_wrap_frame6)*gt_mask6
    gt_diff_frame67=(gt_frame7-gt_wrap_frame7)*gt_mask7

    wrap_frame=torch.cat((wrap_frame2,wrap_frame3,wrap_frame4,wrap_frame5,wrap_frame6,wrap_frame7),2)
    gt_wrap_frame=torch.cat((gt_wrap_frame2,gt_wrap_frame3,gt_wrap_frame4,gt_wrap_frame5,gt_wrap_frame6,gt_wrap_frame7),2)

    diff_frame=torch.cat((diff_frame12,diff_frame23,diff_frame34,diff_frame45,diff_frame56,diff_frame67),2)
    gt_diff_frame=torch.cat((gt_diff_frame12,gt_diff_frame23,gt_diff_frame34,gt_diff_frame45,gt_diff_frame56,gt_diff_frame67),2)
    return diff_frame,gt_diff_frame,wrap_frame,gt_wrap_frame


def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        x=torch.squeeze(x,2)
        flo=torch.squeeze(flo,2)
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        #if x.is_cuda:
        #    grid = grid.cuda()
        vgrid = torch.Tensor(grid).cuda() - flo.cuda()

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)    
        #x=x.cuda()
        output = nn.functional.grid_sample(x, vgrid,mode='bilinear')
        mask = torch.Tensor(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid,mode='bilinear')

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return torch.unsqueeze(output,2),torch.unsqueeze(mask,2)


