import argparse
import os
import warnings
import model.rkccsnet as rkccsnet
import model.mrccsnet as mrccsnet
import model.csnet as csnet
import model.mcan as mcan
from loss import *
import torch.optim as optim
from data_processor import *
from trainer import *
import shutil

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def main():
    global args
    args = parser.parse_args()
    bestBSD_psnr = 0.0
    # setup_seed(1)

    # Create save directory
    samples_dir = os.path.join("./samples", args.exp_name)
    results_dir = os.path.join("./results", args.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        torch.backends.cudnn.benchmark = True

    if args.model == 'rkccsnet':
        model = rkccsnet.RKCCSNet(sensing_rate=args.sensing_rate)
    elif args.model == 'mrccsnet':
        model = mrccsnet.MRCCSNet(sensing_rate=args.sensing_rate)
    elif args.model == 'csnet':
        model = csnet.CSNet(sensing_rate=args.sensing_rate)
    elif args.model == 'mcan':
        model = mcan.NewPyconvRcan(sensing_rate=args.sensing_rate)

    model = model.cuda()
    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [103, 206, 309, 412, 515], gamma=0.5, last_epoch=-1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90, 120, 150,180], gamma=0.25, last_epoch=-1)
    train_loader, test_loader_bsds, test_loader_set5, test_loader_set14 = data_loader(args)

    if args.resume:
        # Load checkpoint model
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        args.start_epoch = checkpoint["epoch"]
        bestBSD_psnr = checkpoint["bestBSD_psnr"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict}
        # Overwrite the pretrained model weights to the current models
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")

    print('\nModel: %s\n'
          'Sensing Rate: %.6f\n'
          'Epoch: %d\n'
          'Initial LR: %f\n'
          % (args.model, args.sensing_rate, args.epochs, args.lr))

    print('Start training')
    for epoch in range(args.start_epoch, args.epochs):
        print('\ncurrent lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        psnr1, ssim1 = valid_bsds(test_loader_bsds, model, criterion)
        print("----------BSDS----------PSNR: %.2f----------SSIM: %.4f" % (psnr1, ssim1))
        psnr2, ssim2 = valid_set5(test_loader_set5, model, criterion)
        print("----------Set5----------PSNR: %.2f----------SSIM: %.4f" % (psnr2, ssim2))
        psnr3, ssim3 = valid_set14(test_loader_set14, model, criterion)
        print("----------Set14----------PSNR: %.2f----------SSIM: %.4f" % (psnr3, ssim3))

        # Automatically save the model with the highest index
        is_best = psnr1 > bestBSD_psnr
        bestBSD_psnr = max(psnr1, bestBSD_psnr)
        torch.save({"epoch": epoch + 1,
                    "bestBSD_psnr": bestBSD_psnr,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == args.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "last.pth.tar"))

    print('Trained finished.')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mcan',
                        choices=['mrccsnet', 'rkccsnet', 'csnet', 'mcan'],
                        help='choose model to train')
    parser.add_argument('--sensing-rate', type=float, default=0.5,
                        choices=[0.50000, 0.25000, 0.12500, 0.06250, 0.03125, 0.015625],
                        help='set sensing rate')
    parser.add_argument('--epochs', default=618, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--block-size', default=32, type=int,
                        metavar='N', help='block size (default: 32)')
    parser.add_argument('--image-size', default=96, type=int,
                        metavar='N', help='image size used for training (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained model',
                        default='save_temp', type=str)
    parser.add_argument('--exp_name', default='mcan5', type=str,
                        help='The directory exp_name')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='start epoch')
    #parser.add_argument('--resume', default='./samples/mcan_cr_625/epoch_562.pth.tar', type=str)
    parser.add_argument('--resume', default='', type=str)

    main()
