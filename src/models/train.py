import torch


def train(model, train_loader, criterion, optimizer, config, DEVICE="cpu"):
    loss_hist = {"train loss": []}

    for epoch in range(1, config.NUM_EPOCHES + 1):
        model.train()

        epoch_train_loss = 0

        for batch_idx, (img, ) in enumerate(train_loader):
            img = img.to(DEVICE)
            labels = labels.to(DEVICE)

            fake_image = model(img)

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        loss_hist["train loss"].append(epoch_train_loss)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("-------------------------------------------------")

    return loss_hist
