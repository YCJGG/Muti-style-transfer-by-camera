import os
import torch
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
from util import *
import scipy.misc
from torch.utils.serialization import load_lua
import time
import torchvision.transforms as transforms
import cv2
import numpy as np
from scipy.misc import imread, imsave,imresize
import argparse
import time
from tqdm import tqdm


parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='images/content',help='path to train')
parser.add_argument('--stylePath',default='images/style',help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7', help='Path to the decoder1')
parser.add_argument('--cuda', default='store_true', help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")

args = parser.parse_args()

try:
    os.makedirs(args.outf)
except OSError:
    pass

wct = WCT(args)


def styleTransfer(contentImg,styleImg,csF):

    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = wct.transform(cF5,sF5,csF,args.alpha)
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(styleImg)
    cF4 = wct.e4(contentImg)
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = wct.transform(cF4,sF4,csF,args.alpha)
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(styleImg)
    cF3 = wct.e3(Im4)
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transform(cF3,sF3,csF,args.alpha)
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(styleImg)
    cF2 = wct.e2(Im3)
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transform(cF2,sF2,csF,args.alpha)
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(Im2)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform(cF1,sF1,csF,args.alpha)
    Im1 = wct.d1(csF1)

    return Im1.data.cpu()


cImg = torch.Tensor()
sImg = torch.Tensor()
csF = torch.Tensor()
csF = Variable(csF)
if(args.cuda):
    cImg = cImg.cuda(args.gpu)
    sImg = sImg.cuda(args.gpu)
    csF = csF.cuda(args.gpu)
    wct.cuda(args.gpu)



	
styleSet = os.listdir('./images/style/style_image/')

resolution = 300
#contentImg = Image.open('./images/content/content.jpg')

cap = cv2.VideoCapture(1)

ToImage_c = Image.new('RGB',(resolution*3,resolution*3+4))
for i in range(len(styleSet)):
        styleImg_ori = Image.open('./images/style/style_image/'+styleSet[i])
        styleImg_r = imresize(styleImg_ori,[resolution-2,resolution-2])
        #ToImage.paste(ToImage_bg2,((i%3)*resolution,(i//3)*resolution))
        styleImg_r = styleImg_r[:,:,(2,1,0)]
        styleImg_r = Image.fromarray(styleImg_r)
        if i >= 4:
            ToImage_c.paste(styleImg_r,(((i+1)%3)*resolution,((i+1)//3)*resolution))
        else:
            ToImage_c.paste(styleImg_r,((i%3)*resolution,(i//3)*resolution))

while(True):
    ret, frame = cap.read()    
    if ret == False:
        break

    contentImg = frame.copy()
    contentImg_r = imresize(contentImg,[resolution,resolution])
    contentImg = transforms.ToTensor()(contentImg_r)
    contentImg = contentImg.unsqueeze(0)
    cImg = Variable(contentImg)
    cImg = cImg.cuda()

    st = time.time()

    ToImage = ToImage_c.copy()
    #ToImage_bg = Image.new('RGB',(resolution//4+2,resolution//4+2),(255,255,255))
    #ToImage_bg2 = Image.new('RGB',(resolution,resolution),(255,255,255))
    

    i = 0
    flag = 0 
    #contentImg_r = contentImg_r[:,:,(2,1,0)]
    ori = Image.fromarray(contentImg_r, 'RGB')
    ToImage.paste(ori,(resolution*1,resolution*1))
    #cv2.imshow('ori',frame)
    key = cv2.waitKey(1) & 0xFF
    #print(frame)
    if key == ord('q'):
        break
    if key == ord('p'):
        flag = 1
        for img_name in tqdm(styleSet):
            styleImg_ori = Image.open('./images/style/style_image/'+img_name)
            styleImg_r = imresize(styleImg_ori,[resolution,resolution])
            #print(styleImg_r)
            styleImg = transforms.ToTensor()(styleImg_r)
            styleImg = styleImg.unsqueeze(0)
            styleImg = Variable(styleImg)
            sImg = styleImg.cuda()

            #print(type(sImg))
            with torch.no_grad():
                out = styleTransfer(cImg,sImg,csF)
            out = out[0].mul(255).clamp(0,255).byte().permute(1, 2, 0).numpy()
            out = out[:,:,(2,1,0)]
            im = Image.fromarray(out, 'RGB')
            if i < 4:  
                loc_im = ((i%3)*resolution,(i//3)*resolution)
            else:
                loc_im = (((i+1)%3)*resolution,((i+1)//3)*resolution)
            ToImage.paste(im,loc_im)
            #sty = imresize(styleImg_ori,[resolution//4,resolution//4])
            #sty = sty[:,:,(2,1,0)]
            #sty = Image.fromarray(sty, 'RGB')``
            #loc_sty =  ((i%3)*resolution,(i//3)*resolution)
            #ToImage.paste(ToImage_bg,loc_sty)
            #ToImage.paste(sty,loc_sty)
            ToImage = np.array(ToImage)
            #ToImage = ToImage[:,:,(2,1,0)]
            cv2.line(ToImage,(i*(3*resolution//len(styleSet)),3*resolution),((i+1)*(3*resolution//len(styleSet)),3*resolution),(185,255,77),30)
            text = str((100/len(styleSet))*(i+1))+'%'
            x = int(3*resolution/len(styleSet)*(i+1))
            cv2.putText(ToImage, text, (x, int(3*resolution)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA) 
            cv2.imshow('style-transfer',ToImage)
            cv2.waitKey(10)
            ToImage = Image.fromarray(ToImage,'RGB')
            if (i+1) == len(styleSet):
                ToImage.paste(im,loc_im)
            i+=1

    ToImage = np.array(ToImage)
    #ToImage = ToImage[:,:,(2,1,0)]
    ToImage = Image.fromarray(ToImage,'RGB')
    ori = Image.fromarray(contentImg_r, 'RGB')
    ToImage.paste(ori,(resolution*1,resolution*1)) 
    ed = time.time()
    ToImage = np.array(ToImage)
    text1 = 'Quit:q'
    text2 = 'Capture:p'
    text3 = 'Continue: space'
    cv2.putText(ToImage, text1, (resolution, int(resolution+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), lineType=cv2.LINE_AA)
    cv2.putText(ToImage, text2, (resolution, int(resolution+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), lineType=cv2.LINE_AA)
    cv2.putText(ToImage, text3, (resolution, int(resolution+60)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), lineType=cv2.LINE_AA)
    
        #cv2.imshow('',ToImage)
    if flag == 0:
        cv2.imshow('style-transfer',ToImage)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    if flag == 1:
        
        cv2.imshow('style-transfer',ToImage)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        
        
    #imsave('./outputs/output.png',ToImage)
    

