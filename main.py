import logging

import torch

import src.config as config
from src.data.dataset import get_dataloader
from src.models.model import Generator, Discriminator
from src.models.train import train
from src.models.loss import Loss
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

    # tensor = torch.rand(2, 3, 256, 256).to(DEVICE)
    # out = G(tensor)
    # print(out.size())

    # out = D(out)
    # print(out.size())

    # Train

    pass

    # Save

    torch.save(G, './models/sr-g.pt')
    torch.save(D, './models/sr-d.pt')

    print("Program has Ended")
