import torchvision
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


to_pil = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
])


def display_image(image_tensor):
    plt.imshow(F.to_pil_image(image_tensor))
    plt.show()


def plot_tensors(tensor_1):
    plt.axis('off')
    img = torchvision.utils.make_grid(tensor_1[:1], nrow=1).squeeze().detach().cpu()
    plt.imshow(img.permute(1, 2, 0).squeeze())
    plt.show()


def plot_sequential(sequence, x_label, y_label):
    plt.plot(sequence)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
