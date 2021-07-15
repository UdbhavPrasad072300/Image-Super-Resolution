# Image-Super-Resolution

Enhancing Image Resolution with SRGAN

To Run: 

```bash
python main.py
```

## What is Run?

Generator Network (SR-ResNet) is first trained using MSE (Mean-Squared Error) Loss then trained as an GAN
with perceptual loss (vgg features) and BCE (Binary Cross Entropy) Loss with a Discriminator Network

Memory Requirements are Heavy for Generator Network, Discriminator Network and VGG19 Network ( >20 GB)

Implemented from Paper: https://arxiv.org/pdf/1609.04802.pdf
