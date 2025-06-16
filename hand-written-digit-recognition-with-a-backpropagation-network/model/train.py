import torch


def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        target = torch.full(pred.shape, -1, device=device, dtype=torch.float32)
        target[range(100), y] = 1.0
        loss = loss_fn(pred, target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
