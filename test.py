from model import Model
from utils import *
import time
import torch.utils.data

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="./data/small/", type=str, help="path of dataset")
parser.add_argument("-s", "--img_size", default=256, type=int, help="size of image")
parser.add_argument("-b", "--batch_size", default=1, type=int, help="batch size")
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

    torch.cuda.synchronize()
    start = time.time()
    output = model(inputs)
    torch.cuda.synchronize()
    end = time.time()
    print("[Batch %d/%d] [Time %.4f]" % (i, len(dataloader), end - start))
    test_result.update(label, output)

precision, recall, F1, accuracy = test_result.get_result()

print(
    "[P: %.4f%%] [R: %.4f%%] [F1: %.4f%%] [Accuracy: %.4f%%]"
    % (100 * precision, 100 * recall, 100 * F1, 100 * accuracy)
)
