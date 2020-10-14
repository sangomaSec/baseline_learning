import torch
import torch.nn as nn
import torch.distributed


class MoCo(nn.Module):

    def __init__(self, base_encoder, momentum=0.999, t=0.07, dim=128, bs=65536, mlp=False):
        """
        :param base_encoder: query_encoder and key_encoder
        :param momentum: MoCo momentum of updating
        :param t: softmax temperature default is 0.07
        :param dim: the feature dimensions
        :param bs: batch size default is 65536
        """
        super(MoCo, self).__init__()

        self.dim = dim
        self.momentum = momentum
        self.t = t
        self.bs = bs

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        """
        in this block we should know the usage of register_buffer,
        besides, we should know the difference between register_buffer and register_parameter.
        for register_buffer will neglect the operations to compute grad and update its value.
        in this method, MoCo do certain not use grad.
        
        If you have parameters in your model, which should be saved and restored in the state_dict, 
        but not trained by the optimizer, you should register them as buffers.
        Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        """
        self.register_buffer("queue", torch.randn(dim, bs))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        """
        could not get the usage of queue_ptr
        """
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if mlp:
            dim_mlp = self.encoder_q.fc.weight_shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _enqueue_and_dequeue(self, keys):
        """
        the function of the samples to dequeue and enqueue
        for that the encoder for the prior samples may be out date
        """
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len

        self.queue_str[0] = ptr

    @torch.no_grad()
    def forward(self, im_q, im_k):
        """
        :param im_q:
        :param im_k:
        :return: log_its, targets
        """
        # compute query features
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            im_k, idx_un_shuffle = self._shuffle_bn(im_k)

            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._un_shuffle_bn(k, idx_un_shuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        log_its = torch.cat([l_pos, l_neg], dim=1)
        log_its /= self.T

        labels = torch.zeros(log_its.shape[0], dtype=torch.long).cuda()

        self._enqueue_and_dequeue(k)

        return log_its, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """update the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _shuffle_bn(self, x):
        """
        Batch shuffle, for making use of BatchNorm
        :return: idx
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_un_shuffle = torch.argsort(idx_shuffle)

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_un_shuffle

    @torch.no_grad()
    def _un_shuffle_bn(self, x, idx_un_shuffle):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_un_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
