import torch.optim as optim
from torch.optim import SGD
import torch
import torch.nn as nn
from utils.conf import get_device
from utils.args import *
from datasets import get_dataset

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

def get_backbone(bone, old_cols=None, x_shape=None):
    from backbone.MNISTMLP import MNISTMLP
    from backbone.MNISTMLP_PNN import MNISTMLP_PNN
    from backbone.ResNet18 import ResNet
    from backbone.ResNet18_PNN import ResNet18_PNN

    if isinstance(bone, MNISTMLP):
        return MNISTMLP_PNN(bone.input_size, bone.output_size, old_cols)
    elif isinstance(bone, ResNet):
        return ResNet18_PNN(bone.num_classes, bone.nf, old_cols, x_shape)
    else:
        raise NotImplementedError('Progressive Neural Networks is not implemented for this backbone')


class Pnn(nn.Module):
    NAME = 'pnn'
    COMPATIBILITY = ['task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Pnn, self).__init__()
        self.loss = loss
        self.args = args
        self.transform = transform
        self.device = get_device()
        self.x_shape = None
        self.nets = [get_backbone(backbone).to(self.device)]
        self.net = self.nets[-1]
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)

        self.soft = torch.nn.Softmax(dim=0)
        self.logsoft = torch.nn.LogSoftmax(dim=0)
        self.dataset = get_dataset(args)
        self.task_idx = 0

    def forward(self, x, task_label):
        if self.x_shape is None:
            self.x_shape = x.shape

        self.nets[task_label].to(self.device)
        out = self.nets[task_label](x)
        self.nets[task_label].cpu()
        return out

    def end_task(self, dataset):
        # instantiate new column
        if self.task_idx == 4:
            return
        self.task_idx += 1
        self.nets[-1].cpu()
        self.nets.append(get_backbone(dataset.get_backbone(), self.nets, self.x_shape).to(self.device))
        self.net = self.nets[-1]
        self.opt = optim.SGD(self.net.parameters(), lr=self.args.lr)

    def observe(self, inputs, labels, not_aug_inputs):
        if self.x_shape is None:
            self.x_shape = inputs.shape

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
