from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1), nn.ReLU(True)]

    for i in range(num_convs - 1):
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1))
        net.append(nn.ReLU(True))
        net.append(nn.Dropout(0.25))

    net.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*net)


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        net = [vgg_block(1, 3, 64),
               vgg_block(1, 64, 128),
               vgg_block(2, 128, 256),
               vgg_block(2, 256, 512),
               vgg_block(2, 512, 512)]
        self.net = nn.Sequential(*net)
        size = opt.img_size // 2 ** 5
        self.output_layer = nn.Sequential(
            nn.Linear(512 * size ** 2, 100),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.net(img)
        out = out.view(out.shape[0], -1)
        out = self.output_layer(out)
        return out
