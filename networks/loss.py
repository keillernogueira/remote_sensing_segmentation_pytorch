import torch
import torch.nn.functional as F
from torch import nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets, mask):
        # print('---before', inputs.shape, targets.shape, mask.shape)
        num_classes = inputs.shape[1]
        targets_mask = targets.masked_select(mask.type(torch.cuda.BoolTensor))
        inputs_mask = inputs.permute((0, 2, 3, 1)).masked_select(mask.unsqueeze(-1).repeat(1, 1, 1, num_classes).
                                                                 type(torch.cuda.BoolTensor)).view(-1, num_classes)
        # print('---after', inputs_mask.shape, targets_mask.shape)
        return self.nll_loss(F.log_softmax(inputs_mask, dim=1), targets_mask)

