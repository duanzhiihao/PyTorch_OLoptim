import torch


class SGD_globLR(torch.optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, momentum=0, dampening=0, weight_decay=0):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.device = self.param_groups[0]['params'][0].device
        self.state['lr'] = torch.tensor([0.0], device=self.device)
        self.state['lr_sqsum'] = torch.tensor([0.0], device=self.device)

    def step(self):
        """Performs a single optimization step.
        """
        inprod = torch.tensor([0.0], device=self.device)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                gt = p.grad.data
                param_state = self.state[p]
                if 'gt_prev' not in param_state:
                    param_state['gt_prev'] = torch.zeros_like(p.data)
                if weight_decay != 0:
                    gt.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(gt).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, gt)
                    gt = buf

                p.data.add_(-self.state['lr'] * gt)
                inprod += (param_state['gt_prev'] * gt).sum()
                param_state['gt_prev'] = gt.clone()
        self.state['lr_sqsum'].add_(inprod*inprod)
        denom = torch.sqrt(self.state['lr_sqsum']) + 1e-8
        self.state['lr'] += (0.1 / denom) * inprod
        self.state['lr'].clamp_(min=0)
        debug = 1


class SGD_cordLR(torch.optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, momentum=0, dampening=0, weight_decay=0):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.device = self.param_groups[0]['params'][0].device
        self.state['lr4lr'] = 0.001

    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                if p.grad is None:
                    continue
                gt = p.grad.data
                param_state = self.state[p]
                if 'lr' not in param_state:
                    param_state['lr'] = torch.zeros_like(p.data)
                    param_state['gt_prev'] = torch.zeros_like(p.data)
                if weight_decay != 0:
                    gt.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(gt).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, gt)
                    gt = buf
                # calculate the gradient of the loss w.r.t. to lr
                param_state['lr'].add_(self.state['lr4lr'] * param_state['gt_prev'] * gt)
                param_state['gt_prev'] = gt.clone()

                p.data.add_(-param_state['lr'] * gt)
        debug = 1
    
    def get_avg_lr(self):
        sum_ = num = 0
        for group in self.param_groups:
            for p in group['params']:
                sum_ += self.state[p]['lr'].sum().cpu().item()
                num += self.state[p]['lr'].numel().cpu().item()
        return sum_ / num