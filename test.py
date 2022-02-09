from model import Model
from utils import *

import torch.utils.data

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="./data/test/", type=str, help="path of dataset")
parser.add_argument("-s", "--img_size", default=256, type=int, help="size of image")
parser.add_argument("-b", "--batch_size", default=8, type=int, help="batch size")
parser.add_argument("-m", "--model_path", default="./model/model.pt", type=str, help="path of model")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

model = Model(opt)

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

test_result = Result()

for i, (inputs, label) in enumerate(dataloader):
    inputs = inputs.to(device)
    label = label.to(device)

    output = model(inputs)
    test_result.update(label, output)

    print("[Batch %d/%d]" % (i, len(dataloader)))

precision, recall, F1, accuracy = test_result.get_result()

print(
    "[P: %.2f%%] [R: %.2f%%] [F1: %.2f%%] [Accuracy: %.2f%%]"
    % (100 * precision, 100 * recall, 100 * F1, 100 * accuracy)
)
