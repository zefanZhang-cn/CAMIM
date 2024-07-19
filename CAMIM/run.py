# coding:utf-8
import torch
import numpy as np
import json
import opennre.encoder, opennre.model, opennre.framework
import sys
import os
import argparse
import random
import logging
from tqdm import tqdm


def set_seed(seed=666):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased',
                    help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='default',
                    help='Checkpoint name')
parser.add_argument('--only_test', action='store_true',
                    help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
                    help='Mask entity mentions')
parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
parser.add_argument('--seed', default=666, type=int, help="random seed, default is 1")  # [1, 49, 1234, 2021, 4321]
# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--train_file', default='', type=str,
                    help='Training data file')
parser.add_argument('--val_file', default='', type=str,
                    help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
                    help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
                    help='Relation to ID file')

# Hyper-parameters
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
                    help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_epoch', default=8, type=int,
                    help='Max number of training epochs')
parser.add_argument('--sample_ratio', default=1.0, type=float,
                    help="only for low resource.")
args = parser.parse_args()


set_seed(args.seed)  # set seed, default is 1
# Some basic settings
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt_path = 'ckpt/{}.pth.tar'.format(args.ckpt)

# text
root_path = '.'
args.train_file = os.path.join(root_path, 'data', 'txt/ours_train.txt')
args.val_file = os.path.join(root_path, 'data', 'txt/ours_val.txt')
args.test_file = os.path.join(root_path, 'data', 'txt/ours_test.txt')

# original image
args.pic_train_file = os.path.join(root_path, 'data',  'img_org/train')
args.pic_val_file = os.path.join(root_path, 'data',  'img_org/val')
args.pic_test_file = os.path.join(root_path, 'data',  'img_org/test')

# caption
args.cap_train_file = os.path.join(root_path, 'data', 'caption/minicpm_train.txt')
args.cap_val_file = os.path.join(root_path, 'data', 'caption/minicpm_val.txt')
args.cap_test_file = os.path.join(root_path, 'data', 'caption/minicpm_test.txt')


# target relations
args.rel2id_file = os.path.join(root_path, 'data', 'ours_rel2id.json')
if not os.path.exists(args.test_file):
    logging.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
    args.test_file = args.val_file
args.metric = 'micro_f1'

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))
id2rel = {v: k for k, v in rel2id.items()}


# Define the sentence encoder
sentence_encoder = opennre.encoder.CAMIM(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    train_pic_path=args.pic_train_file,
    train_cap_path=args.cap_train_file,
    val_path=args.val_file,
    val_pic_path=args.pic_val_file,
    val_cap_path=args.cap_val_file,
    test_path=args.test_file,
    test_pic_path=args.pic_test_file,
    test_cap_path=args.cap_test_file,
    model=model,
    ckpt=ckpt_path,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt='adamw',
    sample_ratio=args.sample_ratio
)

# Train the model
if not args.only_test:
    framework.train_model('micro_f1')

# Val
for loader in [framework.val_loader]:
    framework.load_state_dict(torch.load(ckpt_path)['state_dict'])
    result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = framework.eval_model(
        loader)
    acc_category = correct_category / org_category
    # Print the result
    logging.info('Val set results:\n')
    logging.info('Accuracy: {}\n'.format(result['acc']))
    logging.info('Micro precision: {}\n'.format(result['micro_p']))
    logging.info('Micro recall: {}\n'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))

# Test
for loader in [framework.test_loader]:
    framework.load_state_dict(torch.load(ckpt_path)['state_dict'])
    result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = framework.eval_model(
        loader)
    acc_category = correct_category / org_category
    # Print the result
    logging.info('Test set results:\n')
    logging.info('Accuracy: {}\n'.format(result['acc']))
    logging.info('Micro precision: {}\n'.format(result['micro_p']))
    logging.info('Micro recall: {}\n'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))

