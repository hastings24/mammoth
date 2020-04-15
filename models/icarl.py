from copy import deepcopy
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via iCaRL.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--wd_reg', type=float, required=True,
                        help='L2 regularization applied to the parameters.')
    parser.add_argument('--softmax_temp', type=float, required=True,
                        help='Temperature of the softmax function.')
    return parser


class ICarl(ContinualModel):
    NAME = 'icarl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ICarl, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.current_task = 0

        self.temperature = self.args.softmax_temp
        self.prev_model = None

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()

        feats = self.net.features(x)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_means = None

        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.current_task)
        loss.backward()

        self.opt.step()

        return loss.item()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        if task_idx == 0:
            # Compute loss on the current task
            outputs = self.net(inputs)[:, :ac]
            targets = self.eye[labels][:, :ac]
            loss = self.bce(outputs, targets)
        else:
            # Compute distillation target
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            targets = self.eye[labels][:, pc:ac]
            with torch.no_grad():
                past_outputs = self.prev_model(inputs)[:, :pc]
            past_outputs = torch.sigmoid(past_outputs / self.temperature)
            comb_targets = torch.cat((past_outputs, targets), dim=1)

            # Compute model output
            cur_outputs = self.net(inputs)[:, :ac]

            # Compute distillation loss
            comb_targets = comb_targets.clone().detach()
            loss = self.bce(cur_outputs, comb_targets)

        if self.args.wd_reg:
            loss += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)

        return loss

    def end_task(self, dataset) -> None:
        with torch.no_grad():
            self.fill_buffer(self.buffer, dataset, self.current_task)

        self.prev_model = deepcopy(self.net).eval()
        self.current_task += 1
        self.class_means = None

    def bce(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            p, t, reduction='none').sum(dim=1).mean()

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)

            class_means.append(self.net.features(x_buf).mean(0))
        self.class_means = torch.stack(class_means)

    def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy.
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        mode = self.net.training
        self.net.eval()
        samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)

        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y = self.buffer.get_all_data()

            mem_buffer.empty()
            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y= buf_x[idx], buf_y[idx]
                first = min(_y_x.shape[0], samples_per_class)
                mem_buffer.add_data(
                    examples=_y_x[:first],
                    labels=_y_y[:first]
                )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        # 2.1 Extract all features
        a_x, a_y, a_f = [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))

            a_f.append(self.net.features(x).to('cpu'))
        a_x, a_y, a_f = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f)

        # 2.2 Compute class means
        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y = a_x[idx], a_y[idx]
            feats = a_f[idx]
            mean_feat = feats.mean(0, keepdim=True)

            running_sum = torch.zeros_like(mean_feat)
            i = 0
            while i < samples_per_class and i < feats.shape[0]:
                cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

                idx_min = cost.argmin().item()

                mem_buffer.add_data(
                    examples=_x[idx_min:idx_min + 1].to(self.device),
                    labels=_y[idx_min:idx_min + 1].to(self.device)
                )

                running_sum += feats[idx_min:idx_min + 1]
                feats[idx_min] = feats[idx_min] + 10000
                i += 1

        assert len(mem_buffer.examples) <= mem_buffer.buffer_size

        self.net.train(mode)
