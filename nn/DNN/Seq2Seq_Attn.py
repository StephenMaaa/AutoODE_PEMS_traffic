import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import os
import itertools
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate = 0):
        """
        Args:
            input_dim: the dimension of input sequences.
            hidden_dim: number hidden units.
            num_layers: number of encode layers.
            dropout_rate: recurrent dropout rate.
        """
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional = True, num_layers=self.num_layers, dropout = dropout_rate)
        
        
    def forward(self, source):
        """
        Args:
            source: input tensor(batch_size*input dimension)
        Return:
            outputs: Prediction
            concat_hidden: hidden states
        """
        outputs, hidden = self.lstm(source)
        return outputs, hidden
    
class AttnDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout_rate = 0):
        """
        Args:
            output_dim: the dimension of output sequences.
            hidden_dim: number hidden units.
            num_layers: number of code layers.
            dropout_rate: recurrent dropout rate.
        """
        super(AttnDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.attn = nn.Linear(self.hidden_dim*2 + self.output_dim, 12)
        self.attn_combine = nn.Linear(self.hidden_dim*2 + self.output_dim, self.output_dim)
        # Since the encoder is bidirectional, decoder has double hidden size
        self.lstm = nn.LSTM(output_dim, self.hidden_dim*2, num_layers = self.num_layers, dropout = dropout_rate)
        self.out = nn.Linear(self.hidden_dim*2, self.output_dim)
      
    def forward(self, x, hidden, encoder_outputs):
        
        """
        Args:
            x: prediction from previous prediction.
            hidden: hidden states from previous cell.
        Returns:
            1. prediction for current step.
            2. hidden state pass to next cell.
        """
        # print(x.shape, hidden[0].shape, encoder_outputs.shape)
        attn_weights = F.softmax(self.attn(torch.cat((x[0], hidden[0][0]), 1)), dim =1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1)).transpose(0,1)
       
        x = torch.cat((x[0], attn_applied[0]), 1)
        x = self.attn_combine(x).unsqueeze(0)
        x = F.relu(x)
        
        output, hidden = self.lstm(x, hidden)  
        prediction = self.out(output.float())
        
        return prediction, hidden     
    
class Seq2Seq_Attn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        """
        Args:
            encoder: Encoder object.
            decoder: Decoder object.
            device: 
        """
        super(Seq2Seq_Attn, self).__init__()
        self.encoder = Encoder(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers) 
        self.decoder = AttnDecoder(output_dim = output_dim, hidden_dim = hidden_dim, num_layers = num_layers) 
        self.device = device

    def forward(self, source, target_length):
        """
        Args:
            source: input tensor.
            target_length: forecasting steps.
        Returns:
            total prediction
        """
        # input_length = source.size(0) 
        # batch_size = source.size(1) 
        batch_size = source.size(0) 
        input_length = source.size(2) 
        source = source.permute(2, 0, 1, 3)
        source = source.reshape((input_length, batch_size, -1))
      
        encoder_outputs = torch.zeros(input_length, batch_size, self.encoder.hidden_dim).to(self.device)
        
        outputs = torch.zeros(target_length, batch_size, self.decoder.output_dim).to(self.device)
        encoder_outputs, encoder_hidden = self.encoder(source)
        
        h = torch.cat([encoder_hidden[0][0:self.encoder.num_layers,:,:], encoder_hidden[0][-self.encoder.num_layers:,:,:]], dim=2, out=None).to(device)
        c = torch.cat([encoder_hidden[1][0:self.encoder.num_layers,:,:], encoder_hidden[1][-self.encoder.num_layers:,:,:]], dim=2, out=None).to(device)
        concat_hidden = (h, c)
        decoder_hidden = concat_hidden
       
        decoder_output = torch.zeros((1, batch_size, self.decoder.output_dim), device = self.device)
        
        for t in range(target_length):     
            decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden, encoder_outputs)
            outputs[t] = decoder_output   
        # print(outputs.shape)
        
        outputs = outputs.permute((1, 0, 2)) 
        outputs = outputs.reshape((batch_size, target_length, 3, -1)) 
        outputs = outputs.permute((0, 2, 1, 3)) 
        return outputs

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# from torch.utils import data
# import os
# import itertools
# from torch.autograd import Variable
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import warnings
# warnings.filterwarnings("ignore")


# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate = 0):
#         """
#         Args:
#             input_dim: the dimension of input sequences.
#             hidden_dim: number hidden units.
#             num_layers: number of encode layers.
#             dropout_rate: recurrent dropout rate.
#         """
#         super(Encoder, self).__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional = True, num_layers=self.num_layers, dropout = dropout_rate, batch_first = True)
        
        
#     def forward(self, source):
#         """
#         Args:
#             source: input tensor(batch_size*input dimension)
#         Return:
#             outputs: Prediction
#             concat_hidden: hidden states
#         """
#         outputs, hidden = self.lstm(source)
#         return outputs, hidden
    
# class AttnDecoder(nn.Module):
#     def __init__(self, output_dim, hidden_dim, num_layers, dropout_rate = 0):
#         """
#         Args:
#             output_dim: the dimension of output sequences.
#             hidden_dim: number hidden units.
#             num_layers: number of code layers.
#             dropout_rate: recurrent dropout rate.
#         """
#         super(AttnDecoder, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.num_layers = num_layers
#         self.attn = nn.Linear(self.hidden_dim*2 + self.output_dim, 30)
#         self.attn_combine = nn.Linear(self.hidden_dim*2 + self.output_dim, self.output_dim)
#         # Since the encoder is bidirectional, decoder has double hidden size
#         self.lstm = nn.LSTM(output_dim, self.hidden_dim*2, num_layers = self.num_layers, dropout = dropout_rate, batch_first = True)
#         self.out = nn.Linear(self.hidden_dim*2, self.output_dim)
      
#     def forward(self, x, hidden, encoder_outputs):
        
#         """
#         Args:
#             x: prediction from previous prediction.
#             hidden: hidden states from previous cell.
#         Returns:
#             1. prediction for current step.
#             2. hidden state pass to next cell.
#         """
#         print(x.shape, hidden[0].shape)
#         attn_weights = F.softmax(self.attn(torch.cat((x[0], hidden[0][0]), 1)), dim =1)
#         # print(attn_weights.shape)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1)).transpose(0,1)
       
#         x = torch.cat((x[0], attn_applied[0]), 1)
#         x = self.attn_combine(x).unsqueeze(0)
#         x = F.relu(x)
        
#         output, hidden = self.lstm(x, hidden)  
#         prediction = self.out(output.float())
        
#         return prediction, hidden     
    
# class Seq2Seq_Attn(nn.Module):
#     # def __init__(self, encoder, decoder, device):
#     #     """
#     #     Args:
#     #         encoder: Encoder object.
#     #         decoder: Decoder object.
#     #         device: 
#     #     """
#     #     super(Seq2Seq_Attn, self).__init__()
#     #     self.encoder = encoder
#     #     self.decoder = decoder
#     #     self.device = device 

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
#         """
#         Args:
#             encoder: Encoder object.
#             decoder: Decoder object.
#             device: 
#         """
#         super(Seq2Seq_Attn, self).__init__()
#         self.encoder = Encoder(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers) 
#         self.decoder = AttnDecoder(output_dim = output_dim, hidden_dim = hidden_dim, num_layers = num_layers) 
#         self.device = device

#     def forward(self, source, target_length):
#         """
#         Args:
#             source: input tensor.
#             target_length: forecasting steps.
#         Returns:
#             total prediction
#         """
#         # print(source.shape)
#         batch_size = source.size(0) 
#         input_length = source.size(2) 
#         source = source.permute(0, 2, 1, 3)
#         source = source.reshape((batch_size, input_length, -1))

#         # input_length = source.size(0) 
#         # batch_size = source.size(1) 
      
#         encoder_outputs = torch.zeros(batch_size, input_length, self.encoder.hidden_dim).to(self.device)
        
#         outputs = torch.zeros(batch_size, target_length, self.decoder.output_dim).to(self.device)
#         encoder_outputs, encoder_hidden = self.encoder(source)
        
#         h = torch.cat([encoder_hidden[0][0:self.encoder.num_layers,:,:], encoder_hidden[0][-self.encoder.num_layers:,:,:]], dim=2, out=None).to(device)
#         c = torch.cat([encoder_hidden[1][0:self.encoder.num_layers,:,:], encoder_hidden[1][-self.encoder.num_layers:,:,:]], dim=2, out=None).to(device)
#         concat_hidden = (h, c)
#         print("here")
#         print(h.shape, c.shape)
#         decoder_hidden = concat_hidden
#         print(decoder_hidden[0].shape)
#         decoder_output = torch.zeros((batch_size, 1, self.decoder.output_dim), device = self.device)
        
#         for t in range(target_length):     
#             decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden, encoder_outputs)
#             outputs[t] = decoder_output   
#         return outputs