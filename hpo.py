import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


d_preed_num = 133


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, Y in test_loader:
            output = model(X)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(Y.view_as(pred)).sum().item()

    print(
        "\nAccuracy: {}/{} ({:.0f}%)\n".format(
            correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer):
    model.train()
    trainedd = 0
    num = len(train_loader.dataset)
    for (X, Y) in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        trainedd += len(X)
        loss.backward()
        optimizer.step()
        print(f"{trainedd}/{num} trained")


def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, d_preed_num))

    return model


def create_data_loader(data, transform_functions, batch_size, shuffle=True):
    data = datasets.ImageFolder(data, transform=transform_functions)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    args = parser.parse_args()

    model = net()

    train_transforms = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor()])

    Trainload = create_data_loader(args.train, train_transforms, args.batch_size)
    testload = create_data_loader(args.test, test_transforms, args.test_batch_size, shuffle=False)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model, Trainload, loss_criterion, optimizer)
        test(model, testload)

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    main()
