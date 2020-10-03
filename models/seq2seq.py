# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:34:56 2020

@author: Salah
"""

import torch
from torch.nn import LSTM, Linear, Embedding, Module, Dropout, CrossEntropyLoss
from torch.optim import SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from models.utils import utils
from timeit import default_timer as timer

class Encoder(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, 
                 max_length, dropout_rate, embedding_weights=None):
        
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding_layer = Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        if not(embedding_weights == None):
            self.embedding_layer.load_state_dict({'weight': embedding_weights})

        self.dropout = Dropout(p=dropout_rate)
        self.lstm_layer = LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, 
                               bidirectional=bidirectional, batch_first=True, dropout=dropout_rate)
        
        
    def forward(self, x, x_lengths, initial_state, inference=False, device=torch.device('cuda')):
        
        if initial_state == None:
            initial_state = self.init_hidden_state(x.size(0), device=device)
        
        if inference:
            
            with torch.no_grad():
                
                x = self.dropout(self.embedding_layer(x))
                x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
                x, last_state = self.lstm_layer(x, initial_state)
                ''' we don't need the cell output in the encoder we are only interested
                in the state variable. state is a tuple (hidden_state, cell_state) '''
                
                x, _ = pad_packed_sequence(x, batch_first=True, total_length=self.max_length)
                
                #x.shape = [batch_size, seq_len, hidden_dim * num_directions]
                #last_state[0].shape = [num_layers * num_directions, batch_size, hidden_dim]
                #last_state[1].shape is equal to last_state[0].shape
                
                return x, last_state
            
        else:
           
            x = self.dropout(self.embedding_layer(x))
            x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
            x, last_state = self.lstm_layer(x, initial_state)
            ''' we don't need the cell output in the encoder we are only interested
            in the state variable. state is a tuple (hidden_state, cell_state) '''
            
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=self.max_length)
            
            #x.shape = [batch_size, seq_len, hidden_dim * directions]
            #last_state[0].shape = [num_layers * num_directions, batch_size, hidden_dim]
            #last_state[1].shape is equal to last_state[0].shape
            
            return x, last_state
        
    
    def init_hidden_state(self, batch_size, device=torch.device('cuda')):
        
        return (torch.zeros([self.num_layers*self.num_directions, batch_size, self.hidden_dim], device=device),
                torch.zeros([self.num_layers*self.num_directions, batch_size, self.hidden_dim], device=device))
    
    
class AttDecoder(Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, encoder_num_layers, 
                 max_encoder_length, max_decoder_length, bidirectional_encoder, 
                 dropout_rate, eos_token, num_layers=1, embedding_weights=None):
        
        super(AttDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.directions = 2 if bidirectional_encoder else 1
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        
        self.embedding = Embedding(vocab_size, embedding_dim, padding_idx=0, _weight=embedding_weights)
        self.dropout = Dropout(p=dropout_rate)
        self.lstm_layer = LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.idea_alignment = Linear(hidden_dim*2, hidden_dim)
        self.output = Linear(hidden_dim, vocab_size)
        
    def forward(self, x, x_lengths, initial_state, encoder_outputs, attention, inference=False, device=torch.device('cuda')):
        
        if inference:
            with torch.no_grad():
                x = self.embedding(x)
                #x.shape = [batch_size, 1, embed_dim]
                #the input in inference is the trigger token
                
                context = initial_state[0] + initial_state[1]
                #context.shape = [encoder_num_layers*encoder_num_directions, batch_size, hidden_dim]
                context = context.view(context.size(1), context.size(0), -1)
                #context.shape = [batch_size, encoder_num_layers*encoder_num_directions, hidden_dim]
                context = torch.sum(context, dim=1, keepdim=False)
                #context.shape = [batch_size, hidden_dim]
                
                #encoder_outputs.shape = [batch_size, encoder_seq_len, hidden_dim*num_directions]
                encoder_outputs = encoder_outputs.view(self.directions, encoder_outputs.size(0),
                                                       encoder_outputs.size(1), -1)
                #encoder_outputs.shape = [num_directions, batch_size, encoder_seq_len, hidden_dim]
                encoder_outputs = torch.sum(encoder_outputs, 0)
                #encoder_outputs.shape = [batch_size, encoder_seq_len, hidden_dim]
                
                batch_output = torch.empty([0, self.max_decoder_length, self.vocab_size], device=device)
                
                #we itereate over the batch
                for sample_index in range(x.size(0)):
                    
                    sample_context = context[sample_index].unsqueeze(0).unsqueeze(0)
                    #sample_context.shape = [1, 1, hidden_dim]
                    lstm_state = (initial_state[0][:, sample_index, :].unsqueeze(1),
                                  initial_state[1][:, sample_index, :].unsqueeze(1))
                    #lstm_state[k].shape = [num_layers*directions, 1, hidden_dim]
                    
                    embedded = x[sample_index].unsqueeze(0)
                    #embedded.shape = [1, 1, embed_dim]
                    #embedded is the input at time-step t, here we initialized it with the trigger token
                    
                    sample_output = torch.empty([1, 0, self.vocab_size], device=device)
                    
                    #we iterate over time
                    for time_step in range(self.max_decoder_length):
                        
                        lstm_input = torch.cat((sample_context, embedded), -1)
                        #lstm_input.shape = [1, 1, hidden_dim + embed_dim]
                        
                        lstm_output, lstm_state = self.lstm_layer(lstm_input, lstm_state)
                        #lstm_input.shape = [1, 1, hidden_dim]
                        #lstm_state[k].shape = [num_layers*num_directions, 1, hidden_dim]
                        
                        attention_input = torch.cat((lstm_output, lstm_input), -1)
                        #attention_input.shape = [1, 1, hidden_dim*2 + embed_dim]
                        
                        attention_weights = attention(attention_input)
                        #attention_weights.shape = [1, 1, max_encoder_len]
                        
                        soft_attention_weigths = F.softmax(attention_weights, dim=-1)
                        #soft_attention_weigths.shape = [1, 1, max_encoder_len]
            
                        h_t_bar = torch.bmm(soft_attention_weigths, encoder_outputs[sample_index].unsqueeze(0))
                        #encoder_outputs[sample_index].unsqueeze(0).shape = [1, max_encoder_len, hidden_dim]
                        #h_t_bar.shape = [1, 1, hidden_dim]
                        
                        aligned_ideas = torch.cat((lstm_output, h_t_bar), -1)
                        #aligned_ideas.shape = [1, 1, hidden_dim*2]
                        aligned_ideas = self.idea_alignment(aligned_ideas)
                        #aligned_ideas.shape = [1, 1, hidden_dim]
                        aligned_ideas = torch.sigmoid(aligned_ideas)
                        #aligned_ideas.shape = [1, 1, hidden_dim]
                        
                        out = lstm_output + h_t_bar*aligned_ideas
                        #out.shape = [1, 1, hidden_dim]
                        out = self.output(out)
                        #out.shape = [1, 1, vocab_size]
                        
                        sample_output = torch.cat((sample_output, out), 1)
                        
                        out = F.softmax(out, dim=-1)
                        #out.shape = [1, 1, vocab_size]
                        out = torch.argmax(out, dim=-1, keepdim=False)
                        #out.shape = [1, 1]
                        
                        embedded = self.embedding(out)
                        #embedded.shape = [1, 1, embed_dim]
                        
                    #sample_output.shape = [1, seq_len, vocab_size]
                    
                    batch_output = torch.cat((batch_output, sample_output), 0)
                    
                #batch_output.shape = [batch_size, seq_len, vocab_size]
                
                return batch_output, None, None

        else:
            
            x = self.dropout(self.embedding(x))
            #x.shape = [batch_size, seq_len, embed_dim]
            
            context = initial_state[0] + initial_state[1]
            #context.shape = [encoder_num_layers*encoder_num_directions, batch_size, hidden_dim]
            context = context.view(context.size(1), context.size(0), -1)
            #context.shape = [batch_size, encoder_num_layers*encoder_num_directions, hidden_dim]
            context = torch.sum(context, dim=1, keepdim=True)
            #context.shape = [batch_size, 1, hidden_dim]
            context = context.expand(context.size(0), x.size(1), context.size(2))
            #conext.shape = [batch_size, seq_len, hidden_dim]
            
            lstm_input = torch.cat((context, x), -1)
            #lstm_input.shape = [batch_size, seq_len, hidden_dim + embed_dim]
            packed_lstm_input = pack_padded_sequence(lstm_input, x_lengths, batch_first=True, enforce_sorted=False)
            lstm_output, _ = self.lstm_layer(packed_lstm_input, initial_state)
            lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True, total_length=x.size(1))
            #lstm_output.shape = [batch_size, seq_len, hidden_dim]
            
            #encoder_outputs.shape = [batch_size, max_encoder_len, hidden_dim*num_directions]
            encoder_outputs = encoder_outputs.view(self.directions, encoder_outputs.size(0),
                                                   encoder_outputs.size(1), -1)
            #encoder_outputs.shape = [num_directions, batch_size, max_encoder_len, hidden_dim]
            encoder_outputs = torch.sum(encoder_outputs, 0)
            #encoder_outputs.shape = [batch_size, max_encoder_len, hidden_dim]
            
            attention_input = torch.cat((lstm_output, lstm_input), -1)
            #attention_input.shape = [batch_size, seq_len, hidden_dim*2 + embed_dim]
            
            with torch.no_grad():
                attention_weigths = attention(attention_input)
                #attention_weigths.shape = [batch_size, seq_len, max_encoder_len]
                soft_attention_weigths = F.softmax(attention_weigths, dim=-1)
                #soft_attention_weigths.shape = [batch_size, seq_len, max_encoder_len]
            
            h_t_bar = torch.bmm(soft_attention_weigths, encoder_outputs)
            #h_t_bar.shape = [batch_size, seq_len, hidden_dim]
           
            attention_targets = torch.zeros([x.size(0), lstm_output.size(1)], device=device, dtype=torch.long)
            for t in range(1, lstm_output.size(1), 1):
                p = (encoder_outputs.size(1)//lstm_output.size(1))*t
                attention_targets[:, t] = p
            
            aligned_ideas = torch.cat((lstm_output, h_t_bar), -1)
            #aligned_ideas.shape = [batch_size, seq_len, hidden_dim*2]
            aligned_ideas = self.idea_alignment(aligned_ideas)
            #aligned_ideas.shape = [batch_size, seq_len, hidden_dim]
            aligned_ideas = torch.sigmoid(aligned_ideas)
            #aligned_ideas.shape = [batch_size, seq_len, hidden_dim]
            
            out = lstm_output + h_t_bar*aligned_ideas
            #out.shape = [batch_size, seq_len, hidden_dim]
            out = self.output(out)
            #out.shape = [batch_size, seq_len, vocab_size]
            
            return out, attention_input.detach(), attention_targets.detach()

class Attention(Module):
    def __init__(self, hidden_dim, embedding_dim, max_length):
        super(Attention, self).__init__()
        
        self.attention = Linear(hidden_dim*2 + embedding_dim, max_length, bias=False)
    
    def forward(self, x):
        
        return self.attention(x)
    
class Seq2Seq(Module):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_encoder_seq_len,
                 max_decoder_seq_len, eos_token, num_encoder_layers=1, 
                 num_decoder_layers=1, bidirectional_encoder=False, dropout_rate=0.2,
                 embedding_weights=None):
        
        super(Seq2Seq, self).__init__()
        
        self.use_embedding = not(embedding_weights == None)
        
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_encoder_layers,
                               bidirectional_encoder, max_encoder_seq_len, dropout_rate,
                               embedding_weights=embedding_weights)
        
        self.decoder = AttDecoder(vocab_size, embed_dim, hidden_dim, num_encoder_layers, 
                                  max_encoder_seq_len, max_decoder_seq_len, bidirectional_encoder, 
                                  dropout_rate, eos_token, num_layers=num_decoder_layers, embedding_weights=embedding_weights)
        
    def forward(self, x_encoder, x_decoder, encoder_lengths, decoder_lengths, 
                attention, inference=False, device=torch.device('cuda')):
        
        
        encoder_outputs, decoder_initial_state = self.encoder(x_encoder, encoder_lengths, None,
                                                              inference=inference, device=device)
        y_hat, attention_input, attention_targets = self.decoder(x_decoder, decoder_lengths, decoder_initial_state,
                                                                   encoder_outputs, attention, inference=inference, device=device)
        
        #in inference mode attention_input and attention_targets equal to None
        
        return y_hat, attention_input, attention_targets
    
    def name(self):
        return 'seq2seq'
    

def train(model, attention, optimizer, loss_function, epochs_num, train_set, validation_set, epochs, path,
          metrics, log_every=50, validation_log_every=20, device=torch.device('cuda')):
    
    start = timer()
    
    assert(epochs > epochs_num)
    
    attention_loss_function = CrossEntropyLoss(ignore_index=0)
    attention_optimizer = SGD(attention.parameters(), lr=0.01)
    
    for epoch in range(epochs - epochs_num):
        
        counter = 1
        epoch_start = timer()
        
        for batch in train_set:
            
            x_encoder, x_decoder, y, encoder_lengths, decoder_lengths = batch[0]
            x_encoder = x_encoder.to(device)
            x_decoder = x_decoder.to(device)
            y = y.to(device)
            encoder_lengths = encoder_lengths.to(device)
            decoder_lengths = decoder_lengths.to(device)
            
            x_encoder_raw, y_raw = batch[1]
            
            del(batch)
            
            optimizer.zero_grad()
            
            y_hat, attention_input, attention_targets = model(x_encoder, x_decoder, 
                                                               encoder_lengths, decoder_lengths, 
                                                               attention, device=device)
            
            del(x_encoder)
            del(x_decoder)
            del(encoder_lengths)
            del(decoder_lengths)
            
            loss = 0
            
            for j in range(y.size(1)):
               
                y_hat_step = y_hat[:, j]
                y_step = y[:, j]
                loss = loss + loss_function(y_hat_step, y_step)
                del(y_hat_step)
                del(y_step)

            loss.backward()
            optimizer.step()
            
            batch_loss = loss.detach().cpu() / y_hat.size(1)
            
            del(loss)
            
            attention_loss = 0
            attention_optimizer.zero_grad()
            attention_weights = attention(attention_input)
            
            for t in range(attention_targets.size(1)):
                attention_loss = attention_loss + attention_loss_function(attention_weights[:, t], attention_targets[:, t])
            
            attention_loss.backward()
            attention_optimizer.step()
            
            del(attention_loss)
            del(attention_weights)
            del(attention_input)
            del(attention_targets)
            
            if counter % log_every == 0:
                y_hat = F.softmax(y_hat, dim=-1)
                
                metrics.log('train', x_encoder_raw, y.detach().cpu(), y_raw, 
                            torch.argmax(y_hat.detach().cpu(), -1), batch_loss)
        
                data_dict = {'model': model, 'attention': attention, 
                             'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 
                             'metrics': metrics, 'epoch_num': epochs_num}
                utils.save(data_dict, path)
                del(data_dict)
            
            del(y_hat)
            del(y)
            
            counter += 1
        
        epochs_num += 1
        
        evaluate(model, attention, loss_function, validation_set, metrics, 
                 log_every=validation_log_every, mode='validation', device=device)
        
        time = timer()
        print('Epoch finished: ', epoch + 1, '\tEpoch time: ', time - epoch_start,
              '\tOverall time: ', time - start, '\n\n')
        
    
    print('Training finished. Overall time: ', timer() - start, '\n\n')
    
    return model, optimizer, loss_function, epochs_num + epochs
            

def evaluate(model, attention, loss_function, evaluation_set, metrics, mode='test',
             log_every=21, device=torch.device('cuda')):
    
    assert (mode == 'test' or mode == 'validation')
    
    model.eval()
    #running the model on evaluation mode
    
    start = timer()
    
    counter = 1
    
    for batch in evaluation_set:
        
        x_encoder, x_decoder, y, encoder_lengths, decoder_lengths = batch[0]
        x_encoder = x_encoder.to(device)
        x_decoder = x_decoder.to(device)
        y = y.to(device)
        encoder_lengths = encoder_lengths.to(device)
        decoder_lengths = decoder_lengths.to(device)
        
        x_encoder_raw, y_raw = batch[1]
        
        del(batch)
        
        #for the decoder we only give the start of string token
        x_decoder = x_decoder[:, 0].unsqueeze(1)
        #x_decoder.shape = [batch_size, 1]
        
        y_hat, _, _ = model(x_encoder, x_decoder, encoder_lengths, decoder_lengths, attention, inference=True, device=device)
        
        del(x_encoder)
        del(x_decoder)
        del(encoder_lengths)
        del(decoder_lengths)
        
        loss = 0
        
        steps = y.size(1) if y_hat.size(1) > y.size(1) else y_hat.size(1)
        
        for j in range(steps):
            y_hat_step = y_hat[:, j]
            y_step = y[:, j]
            loss += loss_function(y_hat_step, y_step)
            
            del(y_hat_step)
            del(y_step)
            
        loss = loss.mean()
        batch_loss = loss.detach().cpu() / y_hat.size(1)

        del(loss)        
        if counter % log_every == 0:
            y_hat = F.softmax(y_hat, dim=-1)            

            metrics.log(mode, x_encoder_raw, y.detach().cpu(), y_raw,
                        torch.argmax(y_hat.detach().cpu(), -1), batch_loss)
           
        del(y_hat)
        del(y)
        
        counter += 1
    
    model.train()
    #returning back to training mode
        
    print('Evaluation finished. time: ', timer() - start, 'Mode: ', mode, '\n\n')
    
                        
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        