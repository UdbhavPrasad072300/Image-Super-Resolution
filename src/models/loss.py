import torch.nn as nn
import torch.functional as F
import torchvision


def get_perceptual_loss(recon_x, x):
    return F.mse_loss(recon_x, x)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        self.perceptual_loss = None
        self.content_loss = None

        self.model_layers = torchvision.models.vgg.vgg19(pretrained=True).features
        self.content_layers = ["0", "5", "10", "19", "28"]

        for parameter in self.model_layers.parameters():
            parameter.requires_grad = False

    def forward(self, sr_image, original_image):

        self.perceptual_loss = get_perceptual_loss(sr_image, original_image)
        self.content_loss = self.get_content_loss(sr_image, original_image)

        total_loss = self.content_loss + self.perceptual_loss

        return total_loss

    def get_content_loss(self, recon_x, x):
        return F.mse_loss(self.vgg_forward(recon_x), self.vgg_forward(x))

    def vgg_forward(self, image):
        output = image
        output_layers = []
        for name, module in self.model_layers.named_children():
            output = module(output)
            if name in self.content_layers:
                output_layers.append(output)
        return output_layers
