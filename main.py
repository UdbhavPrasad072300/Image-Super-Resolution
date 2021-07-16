import logging

import torch

import src.config as config
from src.data.dataset import get_dataloader
from src.models.model import Generator, Discriminator
from src.models.train import train_generator, train_SRGAN
from src.models.loss import Loss, get_perceptual_loss
from src.models.test import test_model_inputs
from src.visualization.visualize import plot_sequential, plot_tensors

logging.basicConfig(filename="./logs/app.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device being used: {}".format(DEVICE))

if __name__ == "__main__":
    # Data

    train_folder = "./data/raw/cars_test/cars_test/"
    test_folder = "./data/raw/cars_train/cars_train/"

    train_loader, test_loader = get_dataloader(train_folder, test_folder, batch_size=config.BATCH_SIZE)

    logger.info("Train Dataset Length: {}".format(len(train_loader)))
    logger.info("Test Dataset Length: {}".format(len(test_loader)))

    # Model

    G = Generator().to(DEVICE)
    print(G)

    D = Discriminator().to(DEVICE)
    print(D)

    # Training Objects

    criterion = Loss(DEVICE=DEVICE)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=config.G_LR)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=config.D_LR)

    scaler = torch.cuda.amp.GradScaler()

    # Test Model

    G_out, D_out = test_model_inputs(G, D, DEVICE=DEVICE)

    logger.info("Generator Output Size: {}".format(G_out.size()))
    logger.info("Discriminator Output Size: {}".format(D_out.size()))

    # Train

    G_loss_hist = train_generator(G, train_loader, get_perceptual_loss, G_optimizer, scaler, plot_tensors, config,
                                  DEVICE=DEVICE)

    torch.save(G, './models/sr_resnet-g.pt')

    SRGAN_loss_hist = train_SRGAN(G, D, train_loader, criterion, G_optimizer, D_optimizer, scaler, plot_tensors,
                                  config,
                                  DEVICE=DEVICE)

    # Plot Train Stats

    plot_sequential(G_loss_hist["train loss"], "Epoch", "Train SR-ResNet Loss")

    plot_sequential(G_loss_hist["train g loss"], "Epoch", "Train Generator Loss")
    plot_sequential(G_loss_hist["train d loss"], "Epoch", "Train Discriminator Loss")

    # Save

    torch.save(G, './models/sr-g.pt')
    torch.save(D, './models/sr-d.pt')

    print("Program has Ended")
