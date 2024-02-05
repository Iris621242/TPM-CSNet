import torchvision
import torch


def data_loader(args):
    kwopt = {'num_workers': 8, 'pin_memory': True}
    trn_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ])

    test_set5_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])
    test_set14_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])
    test_set11_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])

    # Transformers for BSDS
    test_bsds_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((480, 320)),
        torchvision.transforms.ToTensor(),
    ])

    trn_dataset = torchvision.datasets.ImageFolder('./BSDS500/train', transform=trn_transforms)
    test_bsds = torchvision.datasets.ImageFolder('./BSDS500/val', transform=test_bsds_transforms)
    test_set5 = torchvision.datasets.ImageFolder('./BSDS500/set5', transform=test_set5_transforms)
    test_set14 = torchvision.datasets.ImageFolder('./BSDS500/set14', transform=test_set14_transforms)
    test_set11 = torchvision.datasets.ImageFolder('./BSDS500/set11', transform=test_set14_transforms)
    bsds_imgs = test_bsds.imgs
    set5_imgs = test_set5.imgs
    set14_imgs = test_set14.imgs
    set11_imgs = test_set11.imgs

    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=False, **kwopt,
                                             drop_last=False)
    test_loader_bsds = torch.utils.data.DataLoader(test_bsds, batch_size=1, shuffle=False, **kwopt, drop_last=False)
    test_loader_set5 = torch.utils.data.DataLoader(test_set5, batch_size=1, shuffle=False, **kwopt, drop_last=False)
    test_loader_set14 = torch.utils.data.DataLoader(test_set14, batch_size=1, shuffle=False, **kwopt, drop_last=False)
    test_loader_set11 = torch.utils.data.DataLoader(test_set11, batch_size=1, shuffle=False, **kwopt, drop_last=False)
    


    return trn_loader, test_loader_bsds, test_loader_set5, test_loader_set14, test_loader_set11, bsds_imgs, set5_imgs, set14_imgs, set11_imgs
