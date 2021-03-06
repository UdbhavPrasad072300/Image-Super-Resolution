import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_perceptual_loss(recon_x, x):
    return F.mse_loss(recon_x, x)


class Loss(nn.Module):
    def __init__(self, DEVICE="cpu"):
        super(Loss, self).__init__()

        self.model_layers = torchvision.models.vgg.vgg19(pretrained=True).features.eval().half().to(DEVICE)
        self.content_layers = ["3",
                               "8",
                               "17",
                               "26",
                               "35"]

        for parameter in self.model_layers.parameters():
            parameter.requires_grad = False

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, sr_image, original_image, real_pred, g_fake_pred, d_fake_pred):

        perceptual_loss = get_perceptual_loss(sr_image, original_image)
        content_loss = self.get_content_loss(sr_image, original_image)

        g_total_loss = ((10 ^ -3) * self.get_adversarial_loss(g_fake_pred, False)) + perceptual_loss + \
                       (0.006 * content_loss)

        d_total_loss = self.get_adversarial_loss(real_pred, True) + self.get_adversarial_loss(d_fake_pred, False)

        return g_total_loss, d_total_loss

    def get_adversarial_loss(self, predictions, real_bool):
        real_bool = torch.zeros_like(predictions) if real_bool else torch.ones_like(predictions)
        return self.bce_loss(predictions, real_bool)

    def get_content_loss(self, recon_x, x):
        total_loss = 0
        for x1, x2 in zip(self.vgg_forward(recon_x), self.vgg_forward(x)):
            total_loss += F.mse_loss(x1, x2)
        return total_loss

    def vgg_forward(self, output_image):
        output_feature_maps = []
        for name, module in self.model_layers.named_children():
            output_image = module(output_image)
            if name in self.content_layers:
                output_feature_maps.append(output_image)
        return output_feature_maps
