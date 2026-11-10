import math
from collections import defaultdict

def calculate_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm


class BaseScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

class WarmUpCosineAnnealingLR:
    def __init__(self, optimizer, warm_up_epoch, max_epoch, lr_min, lr_max):
        self.optimizer = optimizer
        self.warm_up_epoch = warm_up_epoch
        self.max_epoch = max_epoch
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.cur_epoch = 1

        self.step()

    def step(self):
        if self.cur_epoch < self.warm_up_epoch:
            lr = self.cur_epoch / self.warm_up_epoch * self.lr_max
        else:
            lr = (self.lr_min + 0.5*(self.lr_max-self.lr_min)*(1.0+math.cos((self.cur_epoch-self.warm_up_epoch)/(self.max_epoch-self.warm_up_epoch)*math.pi)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.cur_epoch += 1


class AverageMeter:
    def __init__(self):
        self.loss_dict = defaultdict(int)   # memorize the sum of each loss for all data
        self.count = 0

    def update(self, batch_loss_dict, batch_size):
        self.count += batch_size
        for loss_name, loss_val in batch_loss_dict.items():
            self.loss_dict[loss_name] += (loss_val * batch_size)

    def average(self):
        for loss_name, loss_val in self.loss_dict.items():
            self.loss_dict[loss_name] = (loss_val / (self.count + 1e-7))

        return self.loss_dict
