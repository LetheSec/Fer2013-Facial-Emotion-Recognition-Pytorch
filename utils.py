import logging
import os
import random

import numpy as np
import torch.nn.functional as F

from models import *


def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    device = true_labels.device
    true_labels = torch.nn.functional.one_hot(
        true_labels, classes).detach().cpu()
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(
            size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)

        true_dist.scatter_(1, torch.LongTensor(
            index.unsqueeze(1)), confidence)
    return true_dist.to(device)


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_model(arch='ResNet18'):
    if arch == 'ResNet18':
        model = ResNet18()
    elif arch == 'ResNet34':
        model = ResNet34()
    elif arch == 'SENet18':
        model = SENet18()
    elif arch == 'DenseNet':
        model = densenet_cifar()
    elif arch == 'VGG19':
        model = VGG('VGG19')
    elif arch == 'PreActResNet18':
        model = PreActResNet18()
    elif arch == 'PreActResNet34':
        model = PreActResNet34()
    elif arch == 'DLA':
        model = DLA()
    elif arch == 'DPN':
        model = DPN26()
    return model


def random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, logfile='output.log'):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=self.logfile
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


def save_checkpoint(state, epoch, is_best, save_path, save_freq=10):
    filename = os.path.join(save_path, 'checkpoint_' + str(epoch) + '.tar')
    if epoch % save_freq == 0:
        torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_path, 'best_checkpoint.tar')
        torch.save(state, best_filename)
