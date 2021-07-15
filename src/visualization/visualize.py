import torchvision
import matplotlib.pyplot as plt


to_pil = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
])


def plot_tensors(tensor_1):
    plt.axis('off')
    img = torchvision.utils.make_grid(tensor_1[:3], nrow=3).squeeze().detach().cpu()
    plt.imshow(img.permute(1, 2, 0).squeeze())
    plt.show()
    return


def plot_sequential(sequence, x_label, y_label):
    plt.plot(sequence)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
