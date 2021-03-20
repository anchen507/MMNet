import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Net
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr, ssim
import numpy as np
import os
import cv2
import skvideo.measure.strred as _strred


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch MMNet")
parser.add_argument("--save", default='./log', type=str, help="Save path")
parser.add_argument("--resume", default="./../restore_model/lamda_0.02/model1_epoch39.pth.tar", type=str, help="Resume path (default: none)")
parser.add_argument("--dataset_dir", default='./../../data/vimeo_septuplet', type=str, help="train_dataset")
parser.add_argument("--save_dir", default='./result/', type=str, help="save path")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=55, help="Number of epochs to train for")
parser.add_argument("--gpu", default=0, type=int, help="gpu ids (default: 1)")
parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning Rate. Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument("--step", type=int, default=6, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], help="Noise training interval")
parser.add_argument("--val_noiseL", type=float, default=50, help='noise level used on validation set')
parser.add_argument("--patch_size", type=int, default=96, help="Patch size")
parser.add_argument("--input_frame", type=int, default=7, help="input frame,usually odd")
parser.add_argument("--output_frame", type=int, default=7, help="output frame,usually odd")

global opt, model
opt = parser.parse_args()
opt.val_noiseL /= 255.
opt.noise_ival[0] /= 255.
opt.noise_ival[1] /= 255.
torch.cuda.set_device(opt.gpu)
device_ids = [0]
# Normalize noise between [0, 1]
lamda=0.02
str_format = 'im%01d.png'
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def test(net,epoch_state):
    valid_set = ValidSetLoader(opt.dataset_dir, patch_size=opt.patch_size, input_frame=opt.input_frame)
    valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    with open(opt.dataset_dir+'/sep_testlist.txt', 'r') as f:
        train_list = f.read().splitlines()
    psnr_list = []
    ssim_list= []
    tpsnr_list =[]
    tssim_list =[]
    sttred_score_list = []
    avg_time=0
    print('testing........')
    for idx_iter, clean_seq in enumerate(valid_loader):
        #print(np.shape(clean_seq))
        print("processing............",idx_iter)
        clean_seq = Variable(clean_seq).cuda()
        gt=clean_seq[:,:,int(opt.input_frame/2-opt.output_frame/2):int(opt.input_frame/2+opt.output_frame/2),:,:]
        noise = torch.empty_like(clean_seq).normal_(mean=0, std=opt.val_noiseL).to(torch.device('cuda'))
        noisy_clean_seq = clean_seq + noise
        start_time=time.time()
        denoise_seq = net(noisy_clean_seq)
        denoise_time=time.time()-start_time
        
        N,C,L,H,W=np.shape(gt)
        denoise_seq = denoise_seq.view(N,C,L,H,W)
        diff_denoise_seq,diff_gt=video_diff(denoise_seq,gt,7)
        psnr_list.append(psnr(denoise_seq.detach(), gt.detach()))
        ssim_list.append(cal_ssim(denoise_seq,gt))
        tpsnr_list.append(psnr(diff_denoise_seq.detach(), diff_gt.detach()))
        tssim_list.append(cal_ssim(diff_denoise_seq,diff_gt))
        sttred_score_list.append(cal_strred(denoise_seq,gt))
        avg_time=avg_time+denoise_time
        print(train_list[idx_iter])
        mkdir_if_not_exist(opt.save_dir+train_list[idx_iter])
        save_out=denoise_seq.permute(0,2,3,4,1)
        save_out=save_out.cpu().numpy()
        for frame_idx in range(7):
            output_img=save_out[0,frame_idx,...]
            output_img[output_img>1.0]=1.0
            output_img[output_img<0]=0
            plt.imsave(os.path.join(opt.save_dir+train_list[idx_iter], str_format % frame_idx),output_img)
    print('valid PSNR---%f, SSIM---%4f, TPSNR---%4f, TSSIM---%4f,STRRED_SCORE---%4f,  average time ----%f' % (float(np.array(psnr_list).mean()),float(np.array(ssim_list).mean()),float(np.array(tpsnr_list).mean()),float(np.array(tssim_list).mean()),float(np.array(sttred_score_list).mean()),avg_time/idx_iter))
    f=open("test_delta.txt","a+")
    f.write("Epoch: %d, PSNR: %.3f, SSIM---%4f, TPSNR---%4f, TSSIM---%4f, STRRED_SCORE---%4f, run time: %.5f---"%(epoch_state, float(np.array(psnr_list).mean()),float(np.array(ssim_list).mean()),float(np.array(tpsnr_list).mean()),float(np.array(tssim_list).mean()),float(np.array(sttred_score_list).mean()),avg_time/idx_iter)+"\n")
    f.close()

def cal_ssim(video1, video2, window_size=11, size_average=False):
    N,C,L,H,W=np.shape(video1)
    ssim_avg=0
    for i in range(N):
       for j in range(L):
          frame_noisy=video1[i,:,j,:,:]
          frame_noisy=frame_noisy.view(1,C,H,W)
          frame_gt=video2[i,:,j,:,:]
          frame_gt=frame_gt.view(1,C,H,W)
          ssim_frame=ssim(frame_noisy.detach(),frame_gt.detach())
          ssim_avg=ssim_avg+ssim_frame.cpu().numpy()
    return ssim_avg/(N*L)
def cal_strred(video1,video2):
    N,C,L,H,W=np.shape(video1)
    video1=video1.view(C,L,H,W).permute(1,2,3,0).detach().cpu().numpy()
    video2=video2.view(C,L,H,W).permute(1,2,3,0).detach().cpu().numpy()
    video1_gray=np.zeros((L,H,W,1),dtype=np.float)
    video2_gray=np.zeros((L,H,W,1),dtype=np.float)
    for i in range(L):
        video1_gray[i,:,:,:]=np.expand_dims((video1[i,:,:,0]*0.2989+video1[i,:,:,1]*0.5870+video1[i,:,:,2]*0.1140),2)*255
        video2_gray[i,:,:,:]=np.expand_dims(video2[i,:,:,0]*0.2989+video2[i,:,:,1]*0.5870+video2[i,:,:,2]*0.1140,2)*255
    _,strred_score,_=_strred(video1_gray,video2_gray)
    return strred_score
def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path,filename))

def save_images(filepath, img):
    # assert the pixel value range is 0-255
    shape=np.shape(img)
    print(sum(sum(sum(img))))
    print(shape[0]*shape[1]*shape[2])
    if (sum(sum(sum(img)))<(shape[0]*shape[1]*shape[2])):
       img=img*255
    im = Image.fromarray(img.astype('uint8')).convert('RGB')
    im.save(filepath, 'png')


def video_diff(video1,video2,frame_length):
    _,V1_frameb=torch.split(video1,[1,frame_length-1],2)
    V1_framef,_=torch.split(video1,[frame_length-1,1],2)
    diff_video1=V1_frameb-V1_framef
    _,V2_frameb=torch.split(video2,[1,frame_length-1],2)
    V2_framef,_=torch.split(video2,[frame_length-1,1],2)
    diff_video2=V2_frameb-V2_framef
    return diff_video1, diff_video2

def main():
    net = Net()
    net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
    with torch.no_grad():
         net.eval()
         test(net,epoch_state)

if __name__ == '__main__':
    main()

