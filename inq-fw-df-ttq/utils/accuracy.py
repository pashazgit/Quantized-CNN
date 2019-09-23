
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.size(0)
        pred = output.topk(maxk, 1, largest=True, sorted=True)[1]
        correct = pred.eq(target.unsqueeze(dim=1).expand_as(pred)).t()
        res = []
        for k in topk:
            correct_k = correct[:k].float().sum().mul_(100.0 / batch_size)
            res.append(correct_k)
        return res
