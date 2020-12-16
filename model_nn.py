import torch
import torch.nn as nn


def same_padding(kernel):

    pad_val = (kernel - 1) / 2
    if kernel%2 == 0:
        out = (int(pad_val-0.5), int(pad_val+0.5))
    else:
        out = int(pad_val)

    return out


class ResnetBlock(nn.Module):
    def __init__(self, in_filters, out_filters, num_kernels1, num_kernels2):
        super(ResnetBlock, self).__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.num_kernels1 = num_kernels1
        self.num_kernels2 = num_kernels2

        padding = same_padding(self.num_kernels1[0])
        self.zero_pad = nn.ZeroPad2d((0, 0, padding[0], padding[1]))
        self.conv1 = nn.Conv2d(self.in_filters, self.out_filters, (self.num_kernels1[0], self.num_kernels2))
        self.bn1 = nn.BatchNorm2d(self.out_filters)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(self.out_filters, self.out_filters, self.num_kernels1[1],
                               padding=same_padding(self.num_kernels1[1]))
        self.bn2 = nn.BatchNorm2d(self.out_filters)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.out_filters, self.out_filters, self.num_kernels1[2],
                               padding=same_padding(self.num_kernels1[2]))
        self.bn3 = nn.BatchNorm2d(self.out_filters)
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Conv2d(self.in_filters, self.out_filters, (1, self.num_kernels2))
        self.bn_shortcut = nn.BatchNorm2d(self.out_filters)
        self.out_block = nn.ReLU()

    def forward(self, inputs):

        x = self.zero_pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)
        x = torch.add(x, shortcut)
        out_block = self.out_block(x)

        return out_block


class Resnet1D(nn.Module):
    def __init__(self, params=None):
        super(Resnet1D, self).__init__()

        self.n_cnn_filters = params['n_cnn_filters']
        self.n_cnn_kernels = params['n_cnn_kernels']
        self.n_fc_units = params['n_fc_units']
        self.n_classes = params['n_classes']
        self.batch_size = params['batch_size']

        # Resnet Blocks
        self.block1 = ResnetBlock(1, self.n_cnn_filters[0], self.n_cnn_kernels, 16)
        self.block2 = ResnetBlock(self.n_cnn_filters[0], self.n_cnn_filters[1], self.n_cnn_kernels, 1)
        self.block3 = ResnetBlock(self.n_cnn_filters[1], self.n_cnn_filters[2], self.n_cnn_kernels, 1)
        self.block4 = ResnetBlock(self.n_cnn_filters[2], self.n_cnn_filters[2], self.n_cnn_kernels, 1)

        # Flatten
        self.flat = nn.Flatten()

        # FC
        self.fc1 = nn.Linear(self.n_cnn_filters[2] * 65, self.n_fc_units[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.n_fc_units[0], self.n_fc_units[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.n_fc_units[1], self.n_classes)

    def forward(self, inputs):

        out_block1 = self.block1(inputs)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)
        out_block4 = self.block4(out_block3)

        x = self.flat(out_block4)
        x = self.fc1(x)
        x = self.fc2(x)
        outputs = self.fc3(x)

        return outputs
