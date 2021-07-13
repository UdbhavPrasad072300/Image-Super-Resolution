import torch


def train_generator(model, train_loader, criterion, optimizer, config, DEVICE="cpu"):
    loss_hist = {"train loss": []}

    for epoch in range(1, config.NUM_EPOCHES + 1):
        model.train()

        epoch_train_loss = 0

        for batch_idx, (original_img, train_img) in enumerate(train_loader):
            original_img = original_img.to(DEVICE)
            train_img = train_img.to(DEVICE)

            fake_img = model(train_img)

            loss = criterion(fake_img, original_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        loss_hist["train loss"].append(epoch_train_loss)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("-------------------------------------------------")

    return model, loss_hist


def train_SRGAN(generator, discriminator, train_loader, criterion, g_optimizer, d_optimizer, config, DEVICE="cpu"):
    return
