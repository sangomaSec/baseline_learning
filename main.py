# This is a sample Python script.
import time
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0  # check if we should delete this column
        self.avg = 0  # check if we should delete this column
        self.sum = 0  # check if we should delete this column
        self.count = 0  # check if we should delete this column

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fms = '{name} {val' + '} ({avg' + self.fmt + '})'
        return fms.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmt_str = self._get_batch_fmt_str(num_batches)
        self.batch_meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmt_str.format(batch)]
        entries += [str(meter) for meter in self.batch_meters]
        print('\t'.join(entries))

    def _get_batch_fmt_str(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        20,
        [batch_time, data_time, losses, top1, top5],
        prefix='Epoch: [{}]')

    end = time.time()
    print('batch_time is {}'.format(batch_time))
    print('data_time is {}'.format(data_time))
    print('losses is {}'.format(losses))
    print('top1 is {}'.format(top1))
    print('top5 is {}'.format(top5))
    print('progress is {}'.format(progress))
    print('end is {}'.format(end))




