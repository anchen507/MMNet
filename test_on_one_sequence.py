import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Net
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr,ssim
import numpy as np
import os
import cv2
import scipy.io as sio
import skvideo.measure.strred as _strred
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch MMNet")
parser.add_argument("--save", default='./log', type=str, help="Save path")
parser.add_argument("--resume", default="./../restore_model/lamda_0.02/model1_epoch35.pth.tar", type=str, help="Resume path (default: none)")
parser.add_argument("--dataset_dir", default='./../../data/vimeo_septuplet/sequences/00001/0266/', type=str, help="eval path")
parser.add_argument("--save_dir", default='./result', type=str, help="save path")
parser.add_argument("--gpu", default=0, type=int, help="gpu ids (default: 1)")
parser.add_argument("--val_noiseL", type=float, default=20, help='noise level used on validation set')
parser.add_argument("--input_frame", type=int, default=7, help="input frame,usually odd")
parser.add_argument("--output_frame", type=int, default=7, help="output frame,usually odd")


global opt, model
opt = parser.parse_args()
opt.val_noiseL /= 255.
torch.cuda.set_device(opt.gpu)
device_ids = [0]
# Normalize noise between [0, 1]
str_format = 'im%01d.png'
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
def test(net,epoch_state):
    clean_seq = load_video(opt.dataset_dir)
    psnr_list = []
    ssim_list = []
    tpsnr_list = []
    tssim_list = []
    avg_time=0
    print('testing........')
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
    strred_score=cal_strred(denoise_seq,gt)
    print(strred_score)
    avg_time=avg_time+denoise_time
    print('valid PSNR---%f, SSIM---%4f, TPSNR---%4f, TSSIM---%4f, average time ----%f' % (float(np.array(psnr_list).mean()),float(np.array(ssim_list).mean()),float(np.array(tpsnr_list).mean()),float(np.array(tssim_list).mean()),avg_time))
    f=open("test_delta.txt","a+")
    f.write("Epoch: %d, PSNR: %.3f, SSIM---%4f, TPSNR---%4f, TSSIM---%4f, run time: %.5f---"%(epoch_state, float(np.array(psnr_list).mean()),float(np.array(ssim_list).mean()),float(np.array(tpsnr_list).mean()),float(np.array(tssim_list).mean()),avg_time)+"\n")
    f.close()
    mkdir_if_not_exist(opt.save_dir)
    save_out=denoise_seq.permute(0,2,3,4,1)
    save_out=save_out.cpu().numpy()
    for frame_idx in range(7):
        output_img=save_out[0,frame_idx,...]
        output_img[output_img>1.0]=1.0
        output_img[output_img<0]=0
        plt.imsave(os.path.join(opt.save_dir, str_format % frame_idx),output_img)


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


def save_images(filepath, img):
    # assert the pixel value range is 0-255
    shape=np.shape(img)
    print(sum(sum(sum(img))))
    print(shape[0]*shape[1]*shape[2])
    if (sum(sum(sum(img)))<(shape[0]*shape[1]*shape[2])):
       img=img*255
    im = Image.fromarray(img.astype('uint8')).convert('RGB')
    im.save(filepath, 'png')

def load_video(dataset_dir):
    files=os.listdir(dataset_dir)
    files=sorted(files)
    sequences=[]
    for file in files:
        frame = Image.open(dataset_dir + '/' + file)
        frame = np.array(frame, dtype=np.float32)/255.0
        frame = frame.transpose(2,0,1)
        sequences.append(frame)
    sequences = np.stack(sequences, 1)
    sequences = torch.from_numpy(np.ascontiguousarray(sequences))
    sequences = torch.unsqueeze(sequences,0).cuda()
    print(np.shape(sequences))
    return sequences
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

