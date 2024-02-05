from utils import *
import time
import cv2
from torchvision.transforms import ToPILImage
import torchvision
import torch
import os
import imgproc
import numpy 
from thop import profile


def train(train_loader, model, criterion, optimizer, epoch):
    print('Epoch: %d' % (epoch + 1))
    model.train()
    sum_loss = 0
    for inputs, _ in train_loader:
        inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[0], inputs) + criterion(outputs[1], inputs)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss


def valid_bsds(valid_loader, model, criterion):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.                       
            outputs = model(inputs)                    
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(outputs[0], inputs)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)
    

features_in_hook = []
features_out_hook = []

def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)


def valid_set5(valid_loader, model, criterion, hr_path, cs_r):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    time_sum = 0    
    
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.   
            
            layer_name = 'head'
            for (name, module) in model.named_modules():
                if name == layer_name:
                    print('---------')
                    module.register_forward_hook(hook=hook)    
                          
            outputs = model(inputs) ###
            
            print(len(features_out_hook))     

            vis_fea_out = torch.tensor(features_out_hook[0].cpu()).squeeze(0).permute(1, 2, 0).cuda()
            for j in range(64):
                img = vis_fea_out[:, :, j].unsqueeze(0)
                torchvision.utils.save_image(img.data,'./results/headman_no_t/%d.png'% (j), padding=0)
                              
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())           
            sum_psnr += psnr
            sm = ssim(outputs[0], inputs)
            sum_ssim += sm
            
            # Save image
            sr_y_tensor = torch.tensor(outputs[0])
            sr_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=False)
            sr_image_path = os.path.join("./results/all_results/set5/")
            
            #获取图片名称并保存            
            img_name = hr_path[iters].split('/')[-1]
            img_name = img_name.split('.png')[0]
            
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sr_image_path + '{}.png'.format(img_name), sr_image)
            cv2.imwrite(sr_image_path + '%s_ICSR_%s_PSNR_%.5s_SSIM_%.6s.png' % (img_name,cs_r, psnr, sm.item()), sr_image)
            
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)
    
def valid_set14(valid_loader, model, criterion, hr_path, cs_r):
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.   
            #torch.cuda.synchronize() 
            #time_start = time.time()                 
            outputs = model(inputs) ###
            #torch.cuda.synchronize() #
            #time_end = time.time() #
            #time_sum = time_end - time_start #
            #print(f"time = {time_sum:5.4f}")  #                                   
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())           
            sum_psnr += psnr
            sm = ssim(outputs[0], inputs)
            sum_ssim += sm
            
            # Save image
            sr_y_tensor = torch.tensor(outputs[0])
            sr_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=False)
            sr_image_path = os.path.join("./results/all_results/set14/")
            
            #获取图片名称并保存            
            img_name = hr_path[iters].split('/')[-1]
            img_name = img_name.split('.png')[0]
            
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sr_image_path + '{}.png'.format(img_name), sr_image)
            cv2.imwrite(sr_image_path + '%s_ICSR_%s_PSNR_%.5s_SSIM_%.6s.png' % (img_name,cs_r, psnr, sm.item()), sr_image)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)

def valid_set11(valid_loader, model, criterion, hr_path, cs_r):  ###
    sum_psnr = 0
    sum_ssim = 0
    _ssim = SSIM().cuda()
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.cuda())[:, 0, :, :].unsqueeze(1) / 255.                     
            outputs = model(inputs) ###                                   
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())           
            sum_psnr += psnr
            sm = ssim(outputs[0], inputs)
            sum_ssim += sm
            
            # Save image
            sr_y_tensor = torch.tensor(outputs[0])
            sr_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=False)
            sr_image_path = os.path.join("./results/all_results/set11/")
            
            #获取图片名称并保存            
            img_name = hr_path[iters].split('/')[-1]
            img_name = img_name.split('.png')[0]
            
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(sr_image_path + '{}.png'.format(img_name), sr_image)
            cv2.imwrite(sr_image_path + '%s_ICSR_%s_PSNR_%.5s_SSIM_%.6s.png' % (img_name,cs_r, psnr, sm.item()), sr_image)
            
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)