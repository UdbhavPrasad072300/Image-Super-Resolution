import logging

import torch

import src.config as config
from src.data.dataset import get_dataloader
from src.models.model import Generator, Discriminator
from src.models.train import train_generator, train_SRGAN
from src.models.loss import Loss, get_perceptual_loss
from src.models.test import Test_Model_Inputs
from src.visualization.visualize import plot_sequential

logging.basicConfig(filename="./logs/app.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

torch.manual_seed(0)

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

    criterion = Loss()
    G_optimizer = torch.optim.Adam(G.parameters(), lr=config.LR)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=config.LR)

    # Test Model

    G_out, D_out = Test_Model_Inputs(G, D, DEVICE=DEVICE)

    print("Generator Output Size: ", G_out.size())
    print("Discriminator Output Size: ", D_out.size())

    # Train

    G, G_loss_hist = train_generator(G, train_loader, get_perceptual_loss, G_optimizer, config, DEVICE=DEVICE)
    G, D, SRGAN_loss_hist = train_SRGAN(G, D, train_loader, criterion, G_optimizer, D_optimizer, config, DEVICE=DEVICE)

    # Save

    torch.save(G, './models/sr-g.pt')
    torch.save(D, './models/sr-d.pt')

    print("Program has Ended")
