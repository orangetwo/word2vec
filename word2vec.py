# encoding:utf-8
import argparse
import torch
import warnings
from torch import optim

from config import configs
from optimizer import StepLR
from trainer import Trainer
from dataset import DataLoader
from skipgram import SkipGram
# from  import init_logger, logger
# from pyw2v.common.tools import seed_everything
from trainingmonitor import TrainingMonitor
from utils import init_logger, logger, seed_everything

warnings.filterwarnings("ignore")


def run(args):
    # **************************** 加载数据集 ****************************
    logger.info('starting load train data from disk')
    train_dataset = DataLoader(skip_header=False,
                               negative_num=args.negative_sample_num,
                               window_size=args.window_size,
                               data_path=configs['data_path'],
                               vocab_path=configs['vocab_path'],
                               vocab_size=args.vocab_size,
                               min_freq=args.min_freq,
                               shuffle=True,
                               seed=args.seed,
                               sample=args.sample)

    # **************************** 模型和优化器 ***********************
    logger.info("initializing model")
    model = SkipGram(embedding_dim=args.embedd_dim, vocab_size=len(train_dataset.vocab))
    optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate)

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=configs['figure_dir'], arch=args.model,)
    lr_scheduler = StepLR(optimizer=optimizer,lr=args.learning_rate, epochs=args.epochs)

    # **************************** training model ***********************
    logger.info('training model....')
    trainer = Trainer(model=model,
                      vocab=train_dataset.vocab,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      logger=logger,
                      training_monitor=train_monitor,
                      lr_scheduler=lr_scheduler,
                      n_gpu=args.n_gpus,
                      model_save_path=configs['model_save_path'],
                      vector_save_path=configs['pytorch_embedding_path']
                      )
    trainer.train(train_data=train_dataset)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Word2Vec model training')
    parser.add_argument("--model", type=str, default='skip_gram')
    parser.add_argument("--task", type=str, default='training word vector')
    parser.add_argument('--seed', default=2018, type=int,
                        help='Seed for initializing training.')
    parser.add_argument('--resume', default=False, type=bool,
                        help='Choose whether resume checkpoint model')
    parser.add_argument('--embedd_dim', default=300, type=int)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--window_size', default=5, type=int)
    parser.add_argument('--n_gpus', default='0', type=str)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--sample', default=1e-3, type=float)
    parser.add_argument('--negative_sample_num', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.025, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--vocab_size', default=30000000, type=int)
    args = parser.parse_args()
    init_logger(log_file=configs['log_dir'] + (args.model + ".log"))
    logger.info(f"seed is {args.seed}")
    seed_everything(seed=args.seed)
    run(args)


if __name__ == '__main__':
    main()
