
# Channel-aware GAN Inversion for Semantic Communication

This repository contains the implementation of a channel-aware GAN inversion method designed to extract meaningful semantic information from original inputs and map it into a channel-correlated latent space, which can eliminates the necessity for additional channel encoder and decoder.

## Prerequisites

Make sure you have the following dependencies installed:

- Python (>=3.6)
- PyTorch
- NumPy
- Pillow
- imageio
- tqdm
- lpips
- piq
- pytorch-msssim

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/recusant7/GAN_SeCom.git
cd GAN_SeCom
```

2. Download the pre-trained model checkpoint:

```bash
wget https://github.com/seasonSH/SemanticStyleGAN/releases/download/1.0.0/CelebAMask-HQ-512x512.pt -O pretrained/CelebAMask-HQ-512x512.pt
```

## Running the Code

Run the inversion code with the following command:

```bash
python main.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --outdir results/inversion --dataset ./data/examples --size 512 --batch_size 8 --snr_db 15
```

Make sure to adjust the arguments based on your requirements. You can find a description of the available arguments in the script.

## Results
![Reconstructed Images](results/vis.png)
The results, including reconstructed images and log files, will be saved in the specified output directory (`--outdir`). Check the log files for average PSNR, MS-SSIM, and LPIPS.



## Acknowledgments

- This code is based on [SemanticGAN](https://github.com/nv-tlabs/semanticGAN_code) and [SemanticStyleGAN](https://github.com/seasonSH/SemanticStyleGAN/tree/main). We extend our sincere thanks to the authors of these projects for their valuable works.

