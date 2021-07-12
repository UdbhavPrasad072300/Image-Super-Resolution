import torch


def train(model, train_loader, criterion, optimizer, config, DEVICE="cpu"):
    loss_hist = {"train accuracy": [], "train loss": []}

    for epoch in range(1, config.NUM_EPOCHES + 1):
        model.train()

        epoch_train_loss = 0

        y_true_train = []
        y_pred_train = []

        for batch_idx, (img, labels) in enumerate(train_loader):
            img = img.to(DEVICE)
            labels = labels.to(DEVICE)

            preds, teacher_preds = model(img)

            loss = criterion(teacher_preds, preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_train.extend(labels.detach().tolist())

            epoch_train_loss += loss.item()

        loss_hist["train loss"].append(epoch_train_loss)

        total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x == y])
        total = len(y_pred_train)
        accuracy = total_correct * 100 / total

        loss_hist["train accuracy"].append(accuracy)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)
        print("-------------------------------------------------")

    return loss_hist
