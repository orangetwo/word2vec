import torch

from utils import model_device


# 训练包装器
from progressbar import ProgressBar


class Trainer(object):
    def __init__(self, model,
                 epochs,
                 logger,
                 n_gpu,
                 vocab,
                 model_save_path,
                 vector_save_path,
                 optimizer,
                 lr_scheduler,
                 training_monitor=None,
                 verbose=1):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.logger = logger
        self.verbose = verbose
        self.training_monitor = training_monitor
        self.lr_scheduler = lr_scheduler
        self.n_gpu = n_gpu
        self.vocab = vocab
        self.vector_save_path = vector_save_path
        self.model_save_path = model_save_path

        self.model, self.device = model_device(n_gpu, model=self.model)
        self.start_epoch = 1

    def _save_info(self):
        state = {
            'epoch': self.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return state

    def save(self):
        id_word = {value: key for key, value in self.vocab.items()}
        state = self._save_info()
        torch.save(state, self.model_save_path)
        self.logger.info('saving word2vec vector')
        metrix = self.model.v_embedding_matrix.weight.data
        with open(self.vector_save_path, "w", encoding="utf-8") as f:
            if self.device == 'cpu':
                vector = metrix.numpy()
            else:
                vector = metrix.cpu().numpy()
            for i in range(len(vector)):
                if i % 1000 == 0:
                    print(f'saving {i} word vector')
                word = id_word[i]
                s_vec = vector[i]
                s_vec = [str(s) for s in s_vec.tolist()]
                write_line = word + " " + " ".join(s_vec) + "\n"
                f.write(write_line)

    # epoch训练
    def train_epoch(self, train_data):
        pbar = ProgressBar(n_batch=len(train_data))
        train_loss_list = []
        self.model.train()
        assert self.model.training
        train_examples = train_data.make_iter()
        for step, batch in enumerate(train_examples):
            batch = tuple(t.to(self.device) for t in batch)
            pos_u, pos_v, neg_u, neg_v = batch
            self.optimizer.zero_grad()
            loss = self.model(pos_u, pos_v, neg_u, neg_v)
            loss.backward()
            self.optimizer.step()
            pbar.batch_step(batch_idx=step, info={'loss': loss.item()})

            train_loss_list.append(loss.item())
        print(" ")

        result = {'loss': sum(train_loss_list) / len(train_loss_list)}
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        return result

    def train(self, train_data):
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            print(f"Epoch {epoch}/{self.start_epoch + self.epochs - 1}")
            train_log = self.train_epoch(train_data)

            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in train_log.items()])
            self.logger.info(show_info)

            if hasattr(self.lr_scheduler, 'epoch_step'):
                self.lr_scheduler.epoch_step(epoch)

            if self.training_monitor:
                # Output each loss and draw
                self.training_monitor.epoch_step(train_log)
            self.save()

