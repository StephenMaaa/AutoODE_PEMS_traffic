import torch
import torch.nn as nn
import torch.nn.functional as F 
device = "cuda"

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate = 0):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.linear = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout = dropout_rate, batch_first = True)
        
    def forward(self, x): 
        # x = self.linear(x) 
        outputs, hidden = self.lstm(x)
        return outputs, hidden
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout_rate = 0):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers = num_layers, 
                            dropout = dropout_rate, batch_first = True)
        
        self.out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)   
        prediction = self.out(output.float())
        return prediction, hidden   

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers).to(device)
        self.decoder = Decoder(output_dim = output_dim, hidden_dim = hidden_dim, num_layers = num_layers).to(device)
        self.output_dim = output_dim
            
    def forward(self, x, target_length):
        batch_size = x.size(0) 
        input_length = x.size(2) 
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((batch_size, input_length, -1))
        
        output_dim = self.decoder.output_dim
        encoder_output, encoder_hidden = self.encoder(x)

        decoder_output = torch.zeros((batch_size, 1, output_dim), device = device)
        decoder_hidden = encoder_hidden
        
        outputs = []
        for t in range(target_length):  
            decoder_output, decoder_hidden = self.decoder(decoder_output, decoder_hidden)
            outputs.append(decoder_output)
        out = torch.cat(outputs, dim = 1) 
        out = out.reshape((batch_size, target_length, 3, -1))
        out = out.permute((0, 2, 1, 3))
        return out 