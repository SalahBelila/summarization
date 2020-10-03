# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:27:57 2020

@author: Salah
"""
from preprocessing import Preprocessor
from models.seq2seq import Seq2Seq, Attention, train, evaluate
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models.utils.utils import Metrics, SEQ2SEQ
import os
import numpy as np

PATH = os.path.dirname(__file__)
CNN_DATASET = 'cnn_dailymail'
GIGAWORD_DATASET = 'gigaword'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_path = PATH + '\\models\\logs'
chkpt_path = PATH + '\\models\\checkpoints'
embedding_path = PATH + '\\glove'

chosen_dataset = None

torch.set_default_dtype(torch.float32)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

def use_dataset(dataset_name):
    
    assert dataset_name == CNN_DATASET or dataset_name == GIGAWORD_DATASET
    
    global chosen_dataset, log_path, chkpt_path
    
    chosen_dataset = dataset_name
    
    log_path += '\\' + chosen_dataset
    
    chkpt_path += '\\' + chosen_dataset

    return chosen_dataset

def preprocess_data(vocab_size, batch_size, num_workers=0, max_input_len=400,
                    max_target_len=100):
    
    p = Preprocessor(chosen_dataset)

    print('preprocessing started')
    train_set, test_set, validation_set = p.create_data_loaders(vocab_size, batch_size, num_workers=num_workers,
                                                                max_input_len=max_input_len, max_target_len=max_target_len)
    print('preprocessing finished')
    
    return p, train_set, test_set, validation_set

def instantiate_metrics(preprocessor, model_name, enable_visio=True):
    
    metrics = Metrics(preprocessor, model_name, log_path, enable_visio)
    print('metrics instantiated')
    
    return metrics

def create_embedding_matrix(preprocessor, vocab_size, embedding_dim, train_set):
    
    assert embedding_dim == 50 or embedding_dim == 100 or embedding_dim == 200 or embedding_dim == 300
    
    embedding_matrix = torch.rand(vocab_size, embedding_dim)
    with open(embedding_path + '\\glove.6B.' + str(embedding_dim) + 'd.txt', 'rb') as f:
        for l in f:
            line = l.decode('ASCII', 'ignore').split()
            
            word = line[0]
            for i in range(len(preprocessor.filters())):
                word = word.replace(preprocessor.filters()[i], '')

            embedding_vector = torch.from_numpy(np.array(line[1:]).astype(np.float))
            
            if not(word in preprocessor.filters() or word == '') and len(embedding_vector) == 200:
                index = preprocessor.texts_to_sequences([word])[0][0]
                embedding_matrix[index] = embedding_vector
                
        eos_token = preprocessor.texts_to_sequences(['eos'])[0][0]
        sos_token = preprocessor.texts_to_sequences(['sos'])[0][0]
        embedding_matrix[1] = torch.rand(embedding_dim)
        embedding_matrix[eos_token] = torch.rand(embedding_dim)
        embedding_matrix[sos_token] = torch.rand(embedding_dim)
        
    f.close()
    
    return embedding_matrix

def instantiate_model(model_name, vocab_size, embed_dim, hidden_dim, lr, bidirectional_encoder, 
                      max_encoder_len, max_decoder_len, eos_token, device=DEVICE,
                      decoder_num_layers=2, dropout_rate=0.5, embedding_weights=None):
    
    attention = None
    
    if model_name == SEQ2SEQ:
        model = Seq2Seq(vocab_size, embed_dim, hidden_dim, max_encoder_len, max_decoder_len, 
                        eos_token, bidirectional_encoder=bidirectional_encoder, num_decoder_layers=decoder_num_layers,
                        dropout_rate=dropout_rate, embedding_weights=embedding_weights)
        attention = Attention(hidden_dim, embed_dim, max_encoder_len)
    else:
        raise ValueError('wrong value for model_name')
        
    print('model created')
    
    model.to(device)
    if not(attention == None):
        attention.to(device)
    print('model moved to device: ', device)

    optimizer = Adam(model.parameters(), lr=lr)
    print('optimizer created')
    
    loss_function = CrossEntropyLoss(ignore_index=0, reduction='mean')
    print('loss function created')
    
    return model, attention, optimizer, loss_function

def start_training(model, attention, optimizer, loss_function, prev_epochs_num, train_set, validation_set,
                   epochs_num, metrics, log_every, validation_log_every, device=DEVICE):
    
    print('training started')
    
    checkpoint_path = chkpt_path + '\\' + model.name() + '.pth'
    
    model, optimizer, loss_function, total_epochs_num = train(model, attention, optimizer, loss_function, 
                                                        prev_epochs_num, train_set, validation_set, epochs_num, 
                                                        checkpoint_path, metrics, log_every=log_every,
                                                        validation_log_every=validation_log_every, device=device)
    
    return model, optimizer, loss_function, total_epochs_num

    
def start_evaluation(model, attention, loss_function, evaluation_set, metrics, log_every, device=DEVICE):
    
    return evaluate(model, attention, loss_function, evaluation_set, metrics, log_every=log_every)






























