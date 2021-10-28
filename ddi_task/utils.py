import random
import logging
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import numpy as np
import transformers
from transformers import AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer, BasicTokenizer
from model import ddi_Bert


MODEL_CLASSES = {
    'biobert': (BertConfig, ddi_Bert, BertTokenizer),
    'scibert': (BertConfig, ddi_Bert, BertTokenizer),
    'roberta': (RobertaConfig, ddi_Bert, RobertaTokenizer),
    'albert': (AlbertConfig, ddi_Bert, AlbertTokenizer)}

MODEL_PATH_MAP = {
    'biobert': 'dmis-lab/biobert-base-cased-v1.1',
    'scibert': 'allenai/scibert_scivocab_cased',
    'roberta': 'roberta-base',
    'albert': 'albert-xxlarge-v1'}

def get_label(args):
    return [label.strip() for label in open(args.label_file, 'r', encoding='utf-8')]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(labels, preds)
    P = precision_score(labels, preds, average='macro')
    R = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    return {
        'P': P,
        'R': R,
        'acc': acc,
        'f1': f1,}

def compute_metrics(preds, labels ):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


