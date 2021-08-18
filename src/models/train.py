import torch


def train_generator(model, train_loader, criterion, optimizer, scaler, plot_tensors, config, DEVICE="cpu"):
    model.train()

    loss_hist = {"train loss": []}

    for epoch in range(1, config.SRRESNET_EPOCHES + 1):
        epoch_train_loss = 0

        # print(torch.cuda.memory_allocated())

        for batch_idx, (original_img, train_img) in enumerate(train_loader):
            original_img = original_img.to(DEVICE)
            train_img = train_img.to(DEVICE)

            with torch.cuda.amp.autocast():
                fake_img = model(train_img)
                loss = criterion(fake_img, original_img)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

            if batch_idx + 1 == len(train_loader):
                plot_tensors(fake_img.to(original_img.dtype))
                plot_tensors(original_img)
                plot_tensors(train_img)

        epoch_train_loss = epoch_train_loss / len(train_loader)

        loss_hist["train loss"].append(epoch_train_loss)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("-------------------------------------------------")

    return model, loss_hist


def train_SRGAN(generator, discriminator, train_loader, Loss, g_optimizer, d_optimizer, scaler, plot_tensors,
                config,
                DEVICE="cpu"):
    generator.train()
    discriminator.train()

    criterion = Loss(DEVICE=DEVICE)

    loss_hist = {"train g loss": [], "train d loss": []}

    for epoch in range(1, config.GAN_NUM_EPOCHES + 1):
        epoch_g_train_loss = 0
        epoch_d_train_loss = 0

        for batch_idx, (original_img, train_img) in enumerate(train_loader):
            original_img = original_img.to(DEVICE)
            train_img = train_img.to(DEVICE)

            with torch.cuda.amp.autocast():
                fake_img = generator(train_img)
                g_fake_pred = discriminator(fake_img)
                d_fake_pred = discriminator(fake_img.detach())
                real_pred = discriminator(original_img.detach())
                g_loss, d_loss = criterion(fake_img, original_img, real_pred, g_fake_pred, d_fake_pred)

            # print(torch.cuda.max_memory_allocated())

            g_optimizer.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()

            d_optimizer.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)
            scaler.update()

            epoch_g_train_loss += g_loss.item()
            epoch_d_train_loss += d_loss.item()

            if batch_idx + 1 == len(train_loader):
                plot_tensors(fake_img.to(original_img.dtype))
                plot_tensors(original_img)
                plot_tensors(train_img)

        epoch_g_train_loss = epoch_g_train_loss / len(train_loader)
        epoch_d_train_loss = epoch_d_train_loss / len(train_loader)

        loss_hist["train g loss"].append(epoch_g_train_loss)
        loss_hist["train d loss"].append(epoch_d_train_loss)

        print("-------------------------------------------------")
        print("Epoch: {} Generator loss: {:.8f} & Discriminator loss: {:.8f}".format(epoch,
                                                                                     epoch_g_train_loss,
                                                                                     epoch_d_train_loss))
        print("-------------------------------------------------")

    return generator, discriminator, loss_hist
