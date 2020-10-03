# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 23:33:57 2020

@author: Salah
"""

import pickle
import os
from models.utils.visio import Visio

#these are the only accepted model names
SEQ2SEQ = 'seq2seq'

class Metrics:
    def __init__(self, preprocessor, model_name, path, enable_visio):
        
        assert (model_name == SEQ2SEQ)
        
        self.p = preprocessor
        self.path = path + '\\' + model_name
        self.enable_visio = enable_visio
        self.visio = Visio()
        
        if not (os.path.exists(self.path)):
            os.mkdir(self.path)
        if not (os.path.exists(self.path + '\\train')):
            os.mkdir(self.path + '\\train')
        if not (os.path.exists(self.path + '\\test')):
            os.mkdir(self.path + '\\test')
        if not (os.path.exists(self.path + '\\validation')):
            os.mkdir(self.path + '\\validation')
            
        self.train_log_num = len([name for name in os.listdir(self.path + '\\train') if os.path.isfile(self.path + '\\train\\' + name)])
        self.test_log_num = len([name for name in os.listdir(self.path + '\\test') if os.path.isfile(self.path + '\\test\\' + name)])
        self.validation_log_num = len([name for name in os.listdir(self.path + '\\validation') if os.path.isfile(self.path + '\\validation\\' + name)])

    
    def log(self, mode, raw_inputs, targets, raw_targets, predictions, loss):
        
        raw_predictions = self.p.sequences_to_texts(predictions.tolist())
        raw_predictions = [get_true_sequence(rp.split(), '<eos>') for rp in raw_predictions]
        
        rouge_1_recall, rouge_1_precision, f_1 = calculate_rouge_n_score(1, raw_predictions, raw_targets)
        rouge_2_recall, rouge_2_precision, f_2 = calculate_rouge_n_score(2, raw_predictions, raw_targets)
        rouge_3_recall, rouge_3_precision, f_3 = calculate_rouge_n_score(3, raw_predictions, raw_targets)
        rouge_4_recall, rouge_4_precision, f_4 = calculate_rouge_n_score(4, raw_predictions, raw_targets)
        prediction_oov_token_density = calculate_oov_token_density('<unk>', '<eos>', raw_predictions)
        
        to_be_used_for_visualization = {'raw_inputs': raw_inputs, 'raw_targets': raw_targets, 
                                        'raw_predictions': raw_predictions, 'loss': loss,
                                        'rouge-1': (rouge_1_recall, rouge_1_precision, f_1),
                                        'rouge-2': (rouge_2_recall, rouge_2_precision, f_2),
                                        'rouge-3': (rouge_3_recall, rouge_3_precision, f_3),
                                        'rouge-4': (rouge_4_recall, rouge_4_precision, f_4),
                                        'prediction_oov_token_density': prediction_oov_token_density}
        
        self._save_metrics(mode, to_be_used_for_visualization)
        
        if self.enable_visio:
            self.visio.update(mode, to_be_used_for_visualization)
        
    def _save_metrics(self, mode, data_dict):
        
        assert type(data_dict) == dict
        
        log_num = None
        
        if mode == 'train':
            log_num = self.train_log_num
            self.train_log_num += 1
        elif mode == 'test':
            log_num = self.test_log_num
            self.test_log_num += 1
        elif mode == 'validation':
            log_num = self.validation_log_num
            self.validation_log_num += 1
        else:
            raise ValueError('wrong value for argument mode. valid values are "train", "test", "validation"')
        
        directory = self.path + '\\' + mode + '\\' + str(log_num + 1) + '.pkl'
       
        save(data_dict, directory)
        

    def _load_metrics(self, mode, num='last', with_info=True):
        
        if num == 'last':
            if mode == 'train':
                num = self.train_log_num
            elif mode == 'test':
                num = self.test_log_num
            elif mode == 'validation':
                num = self.validation_log_num
            else:
                raise ValueError('wrong value for argument mode. valid values are "train", "test", "validation"')
        else:
            assert type(num) == int
        
        directory = self.path + '\\' + mode + '\\' + str(num) + '.pkl'
        
        return load(directory, with_info=with_info)
    
    def load_all_logged_metrics(self, mode):
        
        num = None
        
        if mode == 'train':
            num = self.train_log_num
        elif mode == 'test':
            num = self.test_log_num
        elif mode == 'validation':
            num = self.validation_log_num
        else:
            raise ValueError('wrong value for argument mode. valid values are "train", "test", "validation"')
        
        metrics = {'raw_inputs': [], 'raw_targets': [], 
                    'raw_predictions': [], 'loss': [],
                    'rouge-1': ([], [], []),
                    'rouge-2': ([], [], []),
                    'rouge-3': ([], [], []),
                    'rouge-4': ([], [], []),
                    'predictions_oov_token_density': []}
        
        info = self._load_metrics(mode)[0]

        for i in range(1, num, 1):
            loaded_metrics = self._load_metrics(mode, num=i, with_info=False)
            metrics['raw_inputs'].append(loaded_metrics['raw_inputs'])
            metrics['raw_targets'].append(loaded_metrics['raw_targets'])
            metrics['raw_predictions'].append(loaded_metrics['raw_predictions'])
            metrics['loss'].append(loaded_metrics['loss'])
            metrics['rouge-1'][0].append(loaded_metrics['rouge-1'][0])
            metrics['rouge-1'][1].append(loaded_metrics['rouge-1'][1])
            metrics['rouge-1'][2].append(loaded_metrics['rouge-1'][2])
            metrics['rouge-2'][0].append(loaded_metrics['rouge-2'][0])
            metrics['rouge-2'][1].append(loaded_metrics['rouge-2'][1])
            metrics['rouge-2'][2].append(loaded_metrics['rouge-2'][2])
            metrics['rouge-3'][0].append(loaded_metrics['rouge-3'][0])
            metrics['rouge-3'][1].append(loaded_metrics['rouge-3'][1])
            metrics['rouge-3'][2].append(loaded_metrics['rouge-3'][2])
            metrics['rouge-4'][0].append(loaded_metrics['rouge-4'][0])
            metrics['rouge-4'][1].append(loaded_metrics['rouge-4'][1])
            metrics['rouge-4'][2].append(loaded_metrics['rouge-4'][2])
            metrics['predictions_oov_token_density'].append(loaded_metrics['prediction_oov_token_density'])
        
        return info, metrics
    
    def show(self, mode):
        self.visio.show(mode)

def calculate_rouge_n_score(n, predictions, targets):
    
    #remember that predictions and targets are batches. that is lists of sequences
    
    assert len(predictions) == len(targets)
    #we make sure they are both of the same batch size
    precision = 0.0
    recall = 0.0
    batch_size = len(predictions)
    for i in range(batch_size):
        
        prediction_seq = remove_duplicates(predictions[i])
        target_seq = remove_duplicates(targets[i])
        prediction_ngrams = create_ngrams(prediction_seq, n)
        num_prediction_ngrams = sum(1 for ngram in prediction_ngrams)
        target_ngrams = create_ngrams(target_seq, n)
        num_target_ngrams = sum(1 for ngram in target_ngrams)
        overlap_count = calculate_overlapping_ngrams(prediction_ngrams, target_ngrams)

        if num_prediction_ngrams > 0:
            precision += overlap_count / num_prediction_ngrams
        if num_target_ngrams > 0:
            recall += overlap_count / num_target_ngrams
        
    precision /= batch_size
    recall /= batch_size
    #we calculate the mean over the batch
    
    f_measure = 0.0
    if precision > 0 and recall > 0:
        f_measure = (2 * precision * recall) / (precision + recall)
        #here we calculate the F Measure also known as the Harmonic Mean
    
    return recall, precision, f_measure


def create_ngrams(sequence, n):
    to_be_returned = []
    for j in range(len(sequence)):
        ngram = []
        if ((n + j) <= len(sequence)):
            for i in range(n):
                ngram.append(sequence[j + i])
            to_be_returned.append(ngram)
        
    return to_be_returned

def get_true_sequence(sequence, eos_token):
    seq = []
    for word in sequence:
        if word == eos_token:
            break
        seq.append(word)
        
    return ' '.join(seq)

def calculate_overlapping_ngrams(predictions_ngrams, targets_ngrams):
    
    overlap_count = 0
    
    for p_ngram in predictions_ngrams:
        for t_ngram in targets_ngrams:
            if t_ngram == p_ngram:
                overlap_count += 1
                
    return overlap_count

def remove_duplicates(sequences):
    to_be_returned = []
    
    for token in sequences:
        if token in to_be_returned:
            continue
        to_be_returned.append(token)
    
    return to_be_returned

def calculate_oov_token_density(oov_token, eos_token, sequences):
    
    batch_oov_token_count = 0
    
    for seq in sequences:
        seq_oov_token_count = 0
        seq_len = 0
        for word in seq.split():
            if word == oov_token:
                seq_oov_token_count += 1
            if word == eos_token:
                break
            seq_len += 1
        
        if seq_len > 0:
            batch_oov_token_count += seq_oov_token_count / seq_len
        
    batch_oov_token_count /= len(sequences)
    
    return batch_oov_token_count
        

def save(data_dict, file_name):
    
    assert type(data_dict) == dict
    assert os.path.exists(os.path.dirname(file_name))
    
    with open(file_name, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
def load(file_name, with_info=True):
    
    data_dict = None
    with open(file_name, 'rb') as f:
        data_dict = pickle.load(f)
    f.close()
    
    info = (data_dict.keys())
    
    if with_info:
        return info, data_dict
    else:
        return data_dict
        
        
        