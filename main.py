import os
import sys
from torch.utils.data import DataLoader
# setting path
import shutil
import math
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from imageio import imwrite, mimwrite
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from models import make_model, DualBranchDiscriminator
from criteria.lpips import lpips
from collections import OrderedDict
from torch import nn
import piq
from pytorch_msssim import ssim, ms_ssim
import logging
import time
import random

def tensor2image(tensor):
    images = tensor.cpu().clamp(-1,1).permute(0,2,3,1).numpy()
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class PowerNormalize(nn.Module):
    def __init__(self, t_pow=1):
        super(PowerNormalize, self).__init__()
        self.t_pow = t_pow

    def forward(self, x, dim=(1, 2)):
        pwr = torch.mean(x ** 2, dim, True)
        return np.sqrt(self.t_pow) * x / torch.sqrt(pwr)


class AWGN_Channel(nn.Module):
    def __init__(self, snr_db):
        super(AWGN_Channel, self).__init__()
        self.change_snr(snr_db)

    def change_snr(self, snr_db):
        self.std = 10**(-0.05*snr_db)

    def forward(self, x):
        noise = torch.randn_like(x)*self.std
        return x+noise



class AverageMeter():
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


class ImageDataset():
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Get a list of all image file paths in the 'train' directory
        self.image_paths = [os.path.join(data_dir, img) for img in sorted(
            os.listdir(data_dir))[0:100] if img.endswith(".jpg") or img.endswith(".png")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_path


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def get_transformation(args):
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform


def calc_lpips_loss(im1, im2):
    img_gen_resize = F.adaptive_avg_pool2d(im1, (256, 256))
    target_img_tensor_resize = F.adaptive_avg_pool2d(im2, (256, 256))
    p_loss = percept(img_gen_resize, target_img_tensor_resize).mean()
    return p_loss




def optimize_latent(args, g_ema, target_img_tensor, batch_size):

    noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
    for noise in noises:
        noise.requires_grad = False
    # initialization
    with torch.no_grad():
        noise_sample = torch.randn(10000, 512, device=device)
        latent_mean = g_ema.style(noise_sample).mean(0)
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)
        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    # Channel
    if args.no_noises:
        optimizer = optim.Adam([latent_in], lr=args.lr)
    else:
        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    latent_path = [latent_in.detach().clone()]
    pbar = tqdm(range(args.step))
    latent_in.requires_grad = True
    for i in pbar:
        optimizer.zero_grad()
        optimizer.param_groups[0]['lr'] = get_lr(float(i)/args.step, args.lr)
        img_gen, _ = g_ema([channel(p_norm(latent_in, dim=(1, 2)))],
                           input_is_latent=True, randomize_noise=False, noise=None)

        # VGG loss
        p_loss = calc_lpips_loss(img_gen, target_img_tensor)
        # L1_loss
        l1_loss = F.mse_loss(img_gen, target_img_tensor)
        # ssim_loss
        ssim_loss = 1 - ms_ssim(img_gen.clip(0, 1)*0.5+0.5,
                             target_img_tensor*0.5+0.5, data_range=1, size_average=True)
        if args.w_plus == True:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.unsqueeze(
                0).repeat(latent_in.size(0), g_ema.n_latent, 1))
        else:
            latent_mean_loss = F.mse_loss(
                latent_in, latent_mean.repeat(latent_in.size(0), 1))

        # main loss function
        loss = (
            p_loss * args.lambda_lpips +
            ssim_loss * args.lambda_ssim +
            l1_loss * args.lambda_l1 +
            latent_mean_loss * args.lambda_mean
        )
        pbar.set_description(
            f' ssim_loss: {ssim_loss.item():.4f} L1 loss: {l1_loss.item():.4f} VGG loss: {p_loss}')

        loss.backward()
        optimizer.step()

        # noise_normalize_(noises)
        latent_path.append(latent_in.detach().clone())

    return latent_path, noises


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    def parse_boolean(x): return not x in ["False", "false", "0"]
    parser.add_argument('--ckpt', type=str, default='pretrained/CelebAMask-HQ-512x512.pt')
    parser.add_argument('--outdir', type=str, default='results/inversion')
    parser.add_argument(
        '--dataset', default="./data/examples")
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--no_noises', type=parse_boolean, default=True)
    parser.add_argument('--w_plus', type=parse_boolean, default=True,
                        help='optimize in w+ space, otherwise w space')
    parser.add_argument('--save_steps', type=parse_boolean, default=False,
                        help='if to save intermediate optimization results')
    parser.add_argument('--truncation', type=float, default=1,
                        help='truncation tricky, trade-off between quality and diversity')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--step', type=int, default=300,
                        help='latent optimization steps')
    parser.add_argument('--noise_regularize', type=float, default=10)
    parser.add_argument('--lambda_l1', type=float, default=0.3)
    parser.add_argument('--lambda_lpips', type=float, default=1)
    parser.add_argument('--lambda_ssim', type=float, default=0)
    parser.add_argument('--lambda_mean', type=float, default=0)
    # chanel snr
    parser.add_argument('--snr_db', type=int, default=15, help='snr in db')
    # seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    args = parser.parse_args()
    
    
    args.outdir = os.path.join(args.outdir, str(args.snr_db)+"dB")
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(os.path.join(args.outdir, 'recon'), exist_ok=True)
    if args.save_steps:
        os.makedirs(os.path.join(args.outdir, 'steps'), exist_ok=True)

    os.makedirs(os.path.join(args.outdir, 'latent'), exist_ok=True)
    if not args.no_noises:
        os.makedirs(os.path.join(args.outdir, 'noise'), exist_ok=True)



    # init logger
    t = time.strftime("%m_%d_%H:%M:%S", time.localtime())
    logger = get_logger(
        f"results/log/{t}-{args.snr_db}db.log")
    logger.info(args)

    logger.info("Loading model ...")
    ckpt = torch.load(args.ckpt)
    g_ema = make_model(ckpt['args'])
    g_ema.to(device)
    g_ema.eval()
    g_ema.load_state_dict(ckpt['g_ema'])
    percept = lpips.LPIPS(net_type='vgg').to(device)
    
    # D
    discriminator = DualBranchDiscriminator(
        args.size, args.size, img_dim=3, seg_dim=13, channel_multiplier=2
    ).to(device)
    discriminator.load_state_dict(ckpt['d'])
    discriminator.eval()
    # Power Normalization and AWGN channel
    p_norm = PowerNormalize(t_pow=1)
    p_norm.cuda()
    channel = AWGN_Channel(snr_db=args.snr_db)
    channel.cuda()

    
    transform = get_transformation(args)
    psnrs = []
    ms_ssims = []
    totlal_lpips = []
    nums = []
   

    test_dataset = ImageDataset(
        args.dataset,
        transform=transform)

    data_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=False)  # type: ignore

    iter_psnr = AverageMeter('Iter psnr')
    iter_msssim = AverageMeter('MS-SSIM')
    iter_lpips = AverageMeter('Lpips')
    dims = []
    for batch_idx, (images, path) in enumerate(data_loader):
        images = images.to(device)
        target_img_tensor = images
        latent_path, noises = optimize_latent(
            args, g_ema, images, images.shape[0])
        with torch.no_grad():
            latent = latent_path[-1]
            latent=channel(p_norm(latent_path[-1],dim=(1,2)))
            img_gen, _ = g_ema([latent], input_is_latent=True,
                               randomize_noise=False, noise=None)
            lpips_img = calc_lpips_loss(img_gen, target_img_tensor)
            img_y = img_gen.clamp(-1, 1)*0.5+0.5
            target_img_tensor = target_img_tensor*0.5+0.5
            psnr_img = piq.psnr(target_img_tensor, img_y)
            ssim_img = ms_ssim(target_img_tensor, img_y, data_range=1)
            # Log and visdom update
            iter_psnr.update(psnr_img, images.size(0))
            iter_msssim.update(ssim_img, images.size(0))
            iter_lpips.update(lpips_img, images.size(0))
            imgs = tensor2image(img_gen)
            for i in range(img_gen.shape[0]):
                img_path = os.path.join(args.outdir, 'recon/', path[i][-9:])
                # print(path[i])
                imwrite(img_path, imgs[i])
    logger.info(f"Avg PSNR: {iter_psnr.avg}")
    logger.info(f"Avg MS-SSIM: {iter_msssim.avg}")
    logger.info(f"Avg Lpips: {iter_lpips.avg}")

