import torch
from torch.nn import functional as F
from datasets import get_dataset
from copy import deepcopy
from utils.args import *
from models.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Learning without Forgetting.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, required=True,
                        help='Temperature of the softmax function.')
    return parser

class LwF(ContinualModel):
    NAME = 'lwf'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(LwF, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=0)
        self.logsoft = torch.nn.LogSoftmax(dim=0)
        self.dataset = get_dataset(args)

    def end_task(self, dataset):
        self.old_net = deepcopy(self.net).eval()

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)

        if self.old_net is not None:
            old_outputs = self.old_net(inputs)
            loss += self.args.alpha * F.kl_div(
                self.logsoft(old_outputs / self.args.softmax_temp).to(self.device),
                self.soft(outputs / self.args.softmax_temp).to(self.device),
                None, None, 'sum')

        loss.backward()
        self.opt.step()

        return loss.item()
