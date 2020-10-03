# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 00:52:13 2020

@author: Salah
"""

from IPython.display import clear_output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Visio:
    def __init__(self):
        
        self.train_data = {
            'raw_inputs': [''],
            'raw_predictions': [''],
            'raw_targets': [''],
            'loss': [],
            'rouge-1': ([], [], []),
            'rouge-2': ([], [], []),
            'rouge-3': ([], [], []),
            'rouge-4': ([], [], []),
            'predictions_oov_token_density': [], 
            }
        self.test_data = {
            'raw_inputs': [''],
            'raw_predictions': [''],
            'raw_targets': [''],
            'loss': [],
            'rouge-1': ([], [], []),
            'rouge-2': ([], [], []),
            'rouge-3': ([], [], []),
            'rouge-4': ([], [], []),
            'predictions_oov_token_density': [], 
            }
        self.validation_data = {
            'raw_inputs': [''],
            'raw_predictions': [''],
            'raw_targets': [''],
            'loss': [],
            'rouge-1': ([], [], []),
            'rouge-2': ([], [], []),
            'rouge-3': ([], [], []),
            'rouge-4': ([], [], []),
            'predictions_oov_token_density': [], 
            }
        
        
    def show(self, mode):
        
        titles = ['Loss', 'ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-4', 'OOV Token Density']
        
        train_fig = make_subplots(rows=6, cols=1,
                            subplot_titles = titles)
        
        test_fig = make_subplots(rows=6, cols=1,
                            subplot_titles = titles)
        
        validation_fig = make_subplots(rows=6, cols=1,
                            subplot_titles = titles)
        
        if mode == 'train':
            train_fig.add_trace(go.Scatter(y=self.train_data['loss']), row=1, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-1'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=2, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-1'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=2, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-1'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=2, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-2'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=3, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-2'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=3, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-2'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=3, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-3'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=4, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-3'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=4, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-3'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=4, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-4'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=5, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-4'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=5, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['rouge-4'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=5, col=1)
            train_fig.add_trace(go.Scatter(y=self.train_data['predictions_oov_token_density'], name='Prediction OOV Token Density'), row=6, col=1)
        
        if mode == 'test':
            test_fig.add_trace(go.Scatter(y=self.test_data['loss']), row=1, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-1'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=2, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-1'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=2, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-1'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=2, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-2'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=3, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-2'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=3, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-2'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=3, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-3'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=4, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-3'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=4, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-3'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=4, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-4'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=5, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-4'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=5, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['rouge-4'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=5, col=1)
            test_fig.add_trace(go.Scatter(y=self.test_data['predictions_oov_token_density'], name='Prediction OOV Token Density'), row=6, col=1)
            
        if mode == 'validation':
            validation_fig.add_trace(go.Scatter(y=self.validation_data['loss']), row=1, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-1'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=2, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-1'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=2, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-1'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=2, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-2'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=3, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-2'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=3, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-2'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=3, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-3'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=4, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-3'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=4, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-3'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=4, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-4'][0], name='Recall', marker_color='rgba(0, 255, 0, 1)'), row=5, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-4'][1], name='Precision', marker_color='rgba(0, 0, 255, 1)'), row=5, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['rouge-4'][2], name='F-Measure', marker_color='rgba(255, 0, 0, 1)'), row=5, col=1)
            validation_fig.add_trace(go.Scatter(y=self.validation_data['predictions_oov_token_density'], name='Prediction OOV Token Density'), row=6, col=1)
            
        train_fig.update_layout(height=1500, title_text='Training')
        test_fig.update_layout(height=1500, title_text='Test')
        validation_fig.update_layout(height=1500, title_text='Validation')
        
        if mode == 'train':
            train_fig.show()
            print('\n------------------Train sample:-------------------\n\n')
            print('Input:\n\n', self.train_data['raw_inputs'][-1][0])
            print('\n\nReference:\n\n', self.train_data['raw_targets'][-1][0])
            print('\n\nSystem:\n\n', self.train_data['raw_predictions'][-1][0])
        
        if mode == 'test':
            test_fig.show()
            print('\n------------------Test Sample:-------------------\n\n')
            print('Input:\n\n', self.test_data['raw_inputs'][-1][0])
            print('\n\nTarget:\n\n', self.test_data['raw_targets'][-1][0])
            print('\n\nSystem:\n\n', self.test_data['raw_predictions'][-1][0])
        
        if mode == 'validation':
            validation_fig.show()
            print('\n------------------Validation Sample:-------------------\n\n')
            print('Input:\n\n', self.validation_data['raw_inputs'][-1][0])
            print('\n\nReference:\n\n', self.validation_data['raw_targets'][-1][0])
            print('\n\nSystem:\n\n', self.validation_data['raw_predictions'][-1][0], '\n\n\n')
           

    def update(self, mode, data):
        if mode == 'train':
            self.train_data['raw_inputs'].append(data['raw_inputs'])
            self.train_data['raw_targets'].append(data['raw_targets'])
            self.train_data['raw_predictions'].append(data['raw_predictions'])
            self.train_data['loss'].append(data['loss'])
            self.train_data['rouge-1'][0].append(data['rouge-1'][0])
            self.train_data['rouge-1'][1].append(data['rouge-1'][1])
            self.train_data['rouge-1'][2].append(data['rouge-1'][2])
            self.train_data['rouge-2'][0].append(data['rouge-2'][0])
            self.train_data['rouge-2'][1].append(data['rouge-2'][1])
            self.train_data['rouge-2'][2].append(data['rouge-2'][2])
            self.train_data['rouge-3'][0].append(data['rouge-3'][0])
            self.train_data['rouge-3'][1].append(data['rouge-3'][1])
            self.train_data['rouge-3'][2].append(data['rouge-3'][2])
            self.train_data['rouge-4'][0].append(data['rouge-4'][0])
            self.train_data['rouge-4'][1].append(data['rouge-4'][1])
            self.train_data['rouge-4'][2].append(data['rouge-4'][2])
            self.train_data['predictions_oov_token_density'].append(data['prediction_oov_token_density'])
        elif mode == 'test':
            self.test_data['raw_inputs'].append(data['raw_inputs'])
            self.test_data['raw_targets'].append(data['raw_targets'])
            self.test_data['raw_predictions'].append(data['raw_predictions'])
            self.test_data['loss'].append(data['loss'])
            self.test_data['rouge-1'][0].append(data['rouge-1'][0])
            self.test_data['rouge-1'][1].append(data['rouge-1'][1])
            self.test_data['rouge-1'][2].append(data['rouge-1'][2])
            self.test_data['rouge-2'][0].append(data['rouge-2'][0])
            self.test_data['rouge-2'][1].append(data['rouge-2'][1])
            self.test_data['rouge-2'][2].append(data['rouge-2'][2])
            self.test_data['rouge-3'][0].append(data['rouge-3'][0])
            self.test_data['rouge-3'][1].append(data['rouge-3'][1])
            self.test_data['rouge-3'][2].append(data['rouge-3'][2])
            self.test_data['rouge-4'][0].append(data['rouge-4'][0])
            self.test_data['rouge-4'][1].append(data['rouge-4'][1])
            self.test_data['rouge-4'][2].append(data['rouge-4'][2])
            self.test_data['predictions_oov_token_density'].append(data['prediction_oov_token_density'])
        else:
            self.validation_data['raw_inputs'].append(data['raw_inputs'])
            self.validation_data['raw_targets'].append(data['raw_targets'])
            self.validation_data['raw_predictions'].append(data['raw_predictions'])
            self.validation_data['loss'].append(data['loss'])
            self.validation_data['rouge-1'][0].append(data['rouge-1'][0])
            self.validation_data['rouge-1'][1].append(data['rouge-1'][1])
            self.validation_data['rouge-1'][2].append(data['rouge-1'][2])
            self.validation_data['rouge-2'][0].append(data['rouge-2'][0])
            self.validation_data['rouge-2'][1].append(data['rouge-2'][1])
            self.validation_data['rouge-2'][2].append(data['rouge-2'][2])
            self.validation_data['rouge-3'][0].append(data['rouge-3'][0])
            self.validation_data['rouge-3'][1].append(data['rouge-3'][1])
            self.validation_data['rouge-3'][2].append(data['rouge-3'][2])
            self.validation_data['rouge-4'][0].append(data['rouge-4'][0])
            self.validation_data['rouge-4'][1].append(data['rouge-4'][1])
            self.validation_data['rouge-4'][2].append(data['rouge-4'][2])
            self.validation_data['predictions_oov_token_density'].append(data['prediction_oov_token_density'])
        
        clear_output(wait=True)
        self.show(mode)
        
    def set_data(self, mode, data):
        if mode == 'train':
            self.train_data = data
        elif mode == 'test':
            self.test_data = data
        elif mode == 'validation':
            self.validation_data = data
        else:
            raise ValueError('wrong value for argument mode.')



























