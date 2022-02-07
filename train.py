from model import Model

import torch.utils.data

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="./data/small/", type=str, help="path of dataset")
parser.add_argument("-s", "--img_size", default=256, type=int, help="size of image")
parser.add_argument("-b", "--batch_size", default=8, type=int, help="batch size")
parser.add_argument("-n", "--epoch", default=1, type=int, help="epoch")
parser.add_argument("-m", "--model_path", default="./model/model.pt", type=str, help="path of model")
parser.add_argument("-l", "--label_path", default="./model/label.txt", type=str, help="path of label")
parser.add_argument("--lr", default=0.0002, type=float, help="adam: learning rate")
parser.add_argument("--b1", default=0.5, type=float, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", default=0.999, type=float, help="adam: decay of first order momentum of gradient")

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


bce_loss = torch.nn.BCELoss()
model = Model(opt)

if cuda:
    model.cuda()
    bce_loss.cuda()

model.apply(weights_init_normal)

# Negative: 0, Positive: 1
dataloader = torch.utils.data.DataLoader(
    ImageFolder(
        opt.dataset,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

for epoch in range(opt.epoch):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    correct = 0
    total = 0
    for i, (imgs, label) in enumerate(dataloader):
        optimizer.zero_grad()

        label = label.float().to(device)

        inputs = imgs.to(device)
        output = model(inputs)

        output = output.reshape(-1)
        loss = bce_loss(output, label)

        loss.backward()
        optimizer.step()

        for j in range(len(output)):
            if output[j] > 0.5 and label[j] == 1:
                TP += 1
            elif output[j] > 0.5 and label[j] == 0:
                FP += 1
            elif output[j] < 0.5 and label[j] == 0:
                TN += 1
            elif output[j] < 0.5 and label[j] == 1:
                FN += 1
            if abs(output[j] - label[j]) < 0.5:
                correct += 1
            total += 1

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = correct / total

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [P: %.2f%%] [R: %.2f%%] [F1: %.2f%%] [Accuracy: %.2f%%]"
            % (epoch, opt.epoch, i, len(dataloader), loss.item(),
               100 * precision, 100 * recall, 100 * F1, 100 * accuracy)
        )

torch.save(model.state_dict(), opt.model_path)
with open(opt.label_path, "w") as f:
    f.write("0: Negative\n1: Positive")
