from model import Model

import torch.utils.data

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="./data/valid/", type=str, help="path of dataset")
parser.add_argument("-s", "--img_size", default=256, type=int, help="size of image")
parser.add_argument("-b", "--batch_size", default=8, type=int, help="batch size")
parser.add_argument("-m", "--model_path", default="./model/model.pt", type=str, help="path of model")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

model = Model(opt)

# load model from pt file
model.load_state_dict(torch.load(opt.model_path))

if cuda:
    model.cuda()

dataloader = torch.utils.data.DataLoader(
    ImageFolder(
        opt.dataset,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=False,
)

TP = 0
FP = 0
TN = 0
FN = 0
correct = 0
total = 0

for i, (imgs, label) in enumerate(dataloader):
    imgs = imgs.to(device)
    label = label.to(device)
    output = model(imgs)

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
    print("[Batch %d/%d]" % (i, len(dataloader)))

precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
F1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
accuracy = correct / total

print(
    "[P: %.2f%%] [R: %.2f%%] [F1: %.2f%%] [Accuracy: %.2f%%]"
    % (100 * precision, 100 * recall, 100 * F1, 100 * accuracy)
)
