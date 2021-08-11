# encoding:utf-8


# 优化相关
class StepLR(object):
    '''
    自定义学习率变化机制
        Example:
        >>> scheduler =StepLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.epoch_step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>     validate(...)
    '''

    def __init__(self, optimizer, lr, epochs):
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs

    def epoch_step(self, epoch):
        new_lr = self.lr * (1.0 - 1.0 * epoch / self.epochs)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
