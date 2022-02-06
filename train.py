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
print(model)

if cuda:
    model.cuda()
    bce_loss.cuda()

model.apply(weights_init_normal)

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
    for i, (imgs, label) in enumerate(dataloader):
        optimizer.zero_grad()

        label = label.float().to(device)

        inputs = imgs.to(device)
        output = model(inputs)
        output = output.reshape(-1)
        loss = bce_loss(output, label)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
            % (epoch, opt.epoch, i, len(dataloader), loss.item())
        )
