import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda
from model import NeuralNetwork, train, test


LOAD_MODEL = True
TRAIN = True


transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x * 2) - 1)
])
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

batch_size = 100
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"


model = NeuralNetwork().to(device)
if LOAD_MODEL:
    model.load_state_dict(torch.load("model.pth", weights_only=True))


if TRAIN:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = pred[0].argmax(0), y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
