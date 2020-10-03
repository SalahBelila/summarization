# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:34:04 2020

@author: Salah
"""
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer
import torch

class Preprocessor:
    
    def __init__(self, dataset_name):
        
        self.name = dataset_name
        self.path = os.path.dirname(__file__) + '\\prepared-datasets\\' + self.name + '.pkl'
        self.eos_token = '<eos>'
        self.sos_token = '<sos>'
        self.tokenizer = None
        if not os.path.exists(self.path):
            print('loading dataset to disk. this may take a minute or two...')
            raw_dataset = self._load_dataset_from_tensorflow()
            ready_to_save_data = self._prepare_dataset_for_saving(raw_dataset)
            self._save_prepared_data(ready_to_save_data)
            print('dataset loaded to disk.')
    
    
    def _load_dataset_from_tensorflow(self):
        
        dataset = tfds.load(self.name, as_supervised=True, split=['train', 'test', 'validation'])
        
        return dataset
    
    
    def _prepare_dataset_for_saving(self, raw_data):
        
        # raw_data is a tuple (train_data, test_data, validation_data)
        
        train_data = self._prepare_data_as_supervised(raw_data[0])
        test_data = self._prepare_data_as_supervised(raw_data[1])
        validation_data = self._prepare_data_as_supervised(raw_data[2])
        
        return train_data, test_data, validation_data
    
    
    def _prepare_data_as_supervised(self, data):
        
        # data is a list_like of (input_text, target_text) elements
        # this function returns a list of (input_text, target_text) elements
        
        to_be_returned = []
        for sample in data:
            input_word_sequence = (self.sos_token + ' ' + sample[0] + ' ' + self.eos_token).numpy().decode('ASCII', 'ignore')
            target_word_sequence = (self.sos_token + ' ' + sample[1] + ' ' + self.eos_token).numpy().decode('ASCII', 'ignore')
            to_be_returned.append((input_word_sequence, target_word_sequence))
        
        return to_be_returned
    
    
    def _save_prepared_data(self, data):
        
        data_holder = self._DataHolder(data)
        self._save(data_holder)
    
    
    def _save(self, data_holder):
        
        with open(self.path, 'wb') as f:
            pickle.dump(data_holder, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        
    
    def _load(self, split):
        
        data_holder = None
        with open(self.path, 'rb') as f:
            data_holder = pickle.load(f)
        f.close()
        
        return data_holder.get_data(split)
        
    
    def load_preprocessed_data(self, split, vocab_size, max_input_len=400, max_target_len=150,
                               filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                               to_lower=True, padding='post'):

        ''' 
        split: a string with value 'train' or 'test' or 'validation'
        vocab_size: the size of vocabulary to be used for tokenization we use the most frequent vocab_size - 1 words.
           
        batch_size: batch size to be used by the DataLoaders during trainning.
           
        max_input_len: the max length of input sequences, all sequences that are
        longer will be excluded from the dataset (along with thier targets!!), default = 400.
            
        max_target_len: the max length of target sequences, all sequences that are
        longer will be excluded from the dataset (along with thier inputs!!), default = 150.
            
        filters: a string of characters to be filtered out from texts, default = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\\t\\n'.
            
        to_lower: whether to convert all text to lower during tokenization or not, default = True.
            
        padding: whether to use padding at the end ('post') or at the beggining of sequence ('pre'), 
        or not use padding at all (None), default = 'post'.
        '''
        
        assert type(split) == str
        
        if (split == 'train'):
            self.tokenizer = Tokenizer(num_words=vocab_size, filters=filters, lower=to_lower,
                                  oov_token='<unk>', split=' ')
        
        data = self._load(split)
        raw_data = data
        
        # ._load() returns either train, test or validation data depending on split
        # we also need the raw data for experiments, comparison and visualization later
        
        data = self._to_2_lists(data)#: (List, List)
        # each data set is converted from list of 2-elements tuples to a tuple of 2 lists
        
        if (split == 'train'):
            self.tokenizer.fit_on_texts(data[0])
        
        ''' we only fit the tokenizer on training inputs
            by 'fit' we mean contructing the vocab and initializing the 
            tokenizer for later use '''
        
        assert not (self.tokenizer == None)
        data = (self.tokenizer.texts_to_sequences(data[0]), 
                self.tokenizer.texts_to_sequences(data[1]))#: a tuple of 2 lists of sequences
        # using the tokenizer, we convert raw text samples into integer sequences
        
        data = self._filter_by_lengths(data, max_input_len, max_target_len)#: tuple of 2 lists of sequences
        ''' we removed long inputs and targets based on:
            max_input_len, max_target_len '''
        
        if not (padding == None):
            data = self._as_supervised_pad_list_of_sequences(data, padding=padding)#: a list of 5 tenors
            
        ''' we replaced the 2 lists of inputs targets with 5 tensors:
            (x_encoder, x_decoder, y, encoder_lengths, decoder_lengths)
            where x_encoder is the input of the encoder, x_decoder is the input 
            of the decoder, y is the target. encoder_lengths and decoder_lengths
            are the lengths of non-padded input sequences for encoder and decoder. 
            we need them for training later. they are used with pack_padded_sequence()
            and pad_packed_sequence() methods in pytorch '''
        
        # finally we create data loaders for training and evaluation loops
        
        return data, raw_data
    
    
    def _to_2_lists(self, data):
        
        inputs = []
        targets = []
        
        for sample in data:
            inputs.append(sample[0])
            targets.append(sample[1])
            
        return inputs, targets
    
    
    def _filter_by_lengths(self, data, max_input_len, max_target_len):
                
        length = len(data[0])
        i = 0
        while i < length:
            if (len(data[0][i]) > max_input_len) or (len(data[1][i]) > max_target_len):
                data[0].pop(i)
                data[1].pop(i)
                length -= 1
                i -= 1
            i += 1
        
        return data
    
    
    def _as_supervised_pad_list_of_sequences(self, data, padding='post'):
        
        inputs = data[0]
        decoder_inputs = [t[:-1] for t in data[1]]
        # we excluded the end of string token from the decoder_inputs
        
        targets = [t[1:] for t in data[1]]
        
        # we excluded the start of string token from the targets
        
        encoder_lengths = torch.LongTensor(self._get_lengths_from_sequences(inputs))
        decoder_lengths = torch.LongTensor(self._get_lengths_from_sequences(targets))
        
        inputs = torch.from_numpy(pad_sequences(inputs, padding=padding)).type(torch.LongTensor)
        targets = torch.from_numpy(pad_sequences(targets, padding=padding)).type(torch.LongTensor)
        decoder_inputs = torch.from_numpy(pad_sequences(decoder_inputs, padding=padding)).type(torch.LongTensor)
        
        to_be_returned = [inputs, decoder_inputs, targets, encoder_lengths, decoder_lengths]
          
        return to_be_returned
        
    
    def _get_lengths_from_sequences(self, sequences):
        
        to_be_returned = []
        for seq in sequences:
            to_be_returned.append(len(seq))
            
        return to_be_returned
    
    
    def _create_data_loader(self, split, vocab_size, batch_size, max_input_len=400, max_target_len=150,
                            filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', to_lower=True,
                            padding='post', shuffle=True, num_workers=6):

        data = self._SummarizationDataset(self, split, vocab_size, max_input_len=max_input_len,
                                          max_target_len=max_target_len,
                                          filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                                          to_lower=True, padding=padding)
        
        data = DataLoader(data, batch_size=batch_size, pin_memory=True, 
                          shuffle=shuffle, num_workers=num_workers)
        
        return data    
    
    def create_data_loaders(self, vocab_size, batch_size, num_workers=6, max_input_len=400, 
                           max_target_len=150, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                           to_lower=True, padding='post'):
        
        start = timer()
        
        train = self._create_data_loader('train', vocab_size, batch_size, num_workers=num_workers,
                                         max_input_len=max_input_len, max_target_len=max_target_len,
                                         filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                                         to_lower=True, padding=padding)
        test = self._create_data_loader('test', vocab_size, batch_size, num_workers=num_workers,
                                        max_input_len=max_input_len, max_target_len=max_target_len,
                                        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                                        to_lower=True, padding=padding)
        validation = self._create_data_loader('validation', vocab_size, batch_size, num_workers=num_workers,
                                          max_input_len=max_input_len, max_target_len=max_target_len,
                                          filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                                          to_lower=True, padding=padding)
        
        print('preprocessing time: ', timer() - start)
        
        return train, test, validation
        
    
    def sequences_to_texts(self, sequences):
        return self.tokenizer.sequences_to_texts(sequences)
    
    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)
        
    def get_eos_token(self):
        return self.texts_to_sequences([[self.eos_token]])
        
    def get_sos_token(self):
        return self.texts_to_sequences([[self.sos_token]])
    
    def filters(self):
        return self.tokenizer.filters
    
    class _DataHolder:
        def __init__(self, data):
            
            self.data = data
        
        def get_data(self, split):
            
            assert type(split) == str
            
            if split == 'train':
                return self.data[0]
            
            if split == 'test':
                return self.data[1]
            
            if split == 'validation':
                return self.data[2]
            
            else:
                raise ValueError('valid values are: "train", "test", "validation".')
        
    
    class _SummarizationDataset(Dataset):
        def __init__(self, preprocessor, split, vocab_size, max_input_len=400,
                                    max_target_len=150, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                                    to_lower=True, padding='post'):
            
            self.data, self.raw_data = preprocessor.load_preprocessed_data(split, vocab_size, 
                                                max_input_len=max_input_len, max_target_len=max_target_len, 
                                                filters=filters, to_lower=to_lower, padding=padding)
            self.len = self.data[0].size(0)
            
        def __getitem__(self, index):
            
            data = (self.data[0][index], self.data[1][index], self.data[2][index],
                    self.data[3][index], self.data[4][index])
            raw_data = self.raw_data[index]
            
            return data, raw_data
        
        def __len__(self):
            
            return self.len
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        