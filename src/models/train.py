import torch


def train_generator(model, train_loader, criterion, optimizer, config, DEVICE="cpu"):
    model = model.to(DEVICE).train()

    loss_hist = {"train loss": []}

    for epoch in range(1, config.SRRESNET_EPOCHES + 1):
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

        epoch_train_loss = epoch_train_loss/len(train_loader)

        loss_hist["train loss"].append(epoch_train_loss)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("-------------------------------------------------")

    return model, loss_hist


def train_SRGAN(generator, discriminator, train_loader, criterion, g_optimizer, d_optimizer, config, DEVICE="cpu"):
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

    loss_hist = {"train g loss": [], "train d loss": []}

    for epoch in range(1, config.GAN_NUM_EPOCHES + 1):
        epoch_g_train_loss = 0
        epoch_d_train_loss = 0

        for batch_idx, (original_img, train_img) in enumerate(train_loader):
            original_img = original_img.to(DEVICE)
            train_img = train_img.to(DEVICE)

            fake_img = generator(train_img)
            fake_pred = discriminator(fake_img)
            real_pred = discriminator(original_img)

            g_loss, d_loss = criterion(fake_img, original_img, real_pred, fake_pred)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            epoch_g_train_loss += g_loss.item()
            epoch_d_train_loss += d_loss.item()

        epoch_g_train_loss = epoch_g_train_loss/len(train_loader)
        epoch_d_train_loss = epoch_d_train_loss/len(train_loader)

        loss_hist["train g loss"].append(epoch_g_train_loss)
        loss_hist["train d loss"].append(epoch_d_train_loss)

        print("-------------------------------------------------")
        print("Epoch: {} Generator loss: {:.8f} & Discriminator loss: (:.8f)".format(epoch,
                                                                                     epoch_g_train_loss,
                                                                                     epoch_d_train_loss))
        print("-------------------------------------------------")

    return generator, discriminator, loss_hist
