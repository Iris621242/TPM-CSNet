import argparse
import warnings
import model.mrccsnet as MRCCSNet
import model.mcan as mcan
from loss import *
from data_processor import *
from trainer import *
import os
import torch
import time
from thop import profile

warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()
    print(args)

    if args.model == 'mrccsnet':
        model = MRCCSNet.MRCCSNet(sensing_rate=args.sensing_rate)
    elif args.model == 'mcan':
        model = mcan.NewPyconvRcan(sensing_rate=args.sensing_rate)
        
        

    model = model.cuda()


    # dic = './trained_model/' + str(args.model) + '_' + str(args.sensing_rate) + '.pth'
    dic = './trained_model/mcan/' + str(args.model) + '_' + str(args.sensing_rate) + '.pth' + '.tar'


    # model.load_state_dict(torch.load(dic))
    criterion = loss_fn

    # Load the super-resolution model weights
    checkpoint = torch.load(dic, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Load model weights `{os.path.abspath(dic)}` successfully.")

    trn_loader, bsds, set5, set14, set11,bsds_img,set5_img,set14_img,set11_img = data_loader(args)
    
    img_pathbsd = []
    for i in range(len(bsds_img)):
        j=0
        temp = bsds_img[i][j]
        img_pathbsd.append(temp)

 
    img_path5 = []
    for i in range(len(set5_img)):
        j=0
        temp = set5_img[i][j]
        img_path5.append(temp)


    img_path14 = []
    for i in range(len(set14_img)):
        j=0
        temp = set14_img[i][j]
        img_path14.append(temp)

    
    img_path11 = []
    for i in range(len(set11_img)):
        j=0
        temp = set11_img[i][j]
        img_path11.append(temp)
     
    
     
    # Start the verification mode of the model.
    model.eval()
    '''
    input = torch.randn(1, 1, 256,256)
    flops, params = profile(model, inputs=(input.cuda(), ))##
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6)) ##
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    
    psnr1, ssim1 = valid_bsds(bsds, model, criterion)
    print("----------BSDS----------PSNR: %.2f----------SSIM: %.4f" % (psnr1, ssim1))'''
    psnr2, ssim2 = valid_set5(set5, model, criterion, img_path5, args.sensing_rate)
    print("----------Set5----------PSNR: %.2f----------SSIM: %.4f" % (psnr2, ssim2))
    
    
    '''
    psnr3, ssim3 = valid_set14(set14, model, criterion, img_path14, args.sensing_rate)
    print("----------Set14----------PSNR: %.2f----------SSIM: %.4f" % (psnr3, ssim3))

    img_name = os.listdir(hr_path)
    img_path = []
    for i in img_name:
        temp = os.path.join(hr_path, i)
        img_path.append(temp)
      
    psnr4, ssim4 = valid_set11(set11, model, criterion, img_path11, args.sensing_rate) ###
    print("----------Set11----------PSNR: %.2f----------SSIM: %.4f" % (psnr4, ssim4)) ###
    '''



if __name__ == '__main__':
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rkccsnet',
                        help='choose model to train')
    parser.add_argument('--sensing-rate', type=float, default=0.50000,
                        choices=[0.50000, 0.25000, 0.12500, 0.06250, 0.03125, 0.015625],
                        help='set sensing rate')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--block-size', default=32, type=int,
                        metavar='N', help='block size (default: 32)')
    parser.add_argument('--image-size', default=96, type=int,
                        metavar='N', help='image size used for training (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)

    main()
