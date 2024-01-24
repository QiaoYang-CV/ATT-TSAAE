# -*- Coding: UTF-8 -*-
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import time
import math
import pprint
from matplotlib import pyplot
import torch.nn.functional as F
from torch.autograd import Variable
import math
import convlstm
import TCN
from sklearn.model_selection import train_test_split
import datetime
import os
import timefeatures
from sklearn.preprocessing import StandardScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(0)
np.random.seed(0)


calculate_loss_over_all_values = True

input_window = 32 #50
output_window = 5
batch_size = 50  # batch size=50
multi_features = 3
d_model = 512
predic_step = 10 #10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_data_by_n_days_traindata(series, n, index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = []
    index = -1
    for k in range(n, len(series)-predic_step):
        df.append([])
        index = index + 1

        for i in range(n - 1, -1, -1):
            df[index].append(series[k - i])

        for i in range(n-predic_step,-predic_step,-1):
            df[index].append(series[k - i])
    return df

def generate_data_by_n_days_testdata(series, n, index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = []
    k = 0
    index = -1
    while k+n+predic_step < len(series):
        df.append([])
        index = index + 1
        for i in range(n):
            df[index].append(series[k + i])

        for i in range(predic_step,n+predic_step):
            df[index].append(series[k + i])
        k = k + n
    return df


def multi_generate_traindata_by_n_days(series, n, multi,index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))

    df = []
    index = -1
    for k in range(n, len(series)-predic_step):
        df.append([])
        index = index + 1
        for j in range(multi):
            df[index].append([])
            for i in range(n-1, -1, -1):
                df[index][j].append(series[k - i][j])
            for i in range(n-predic_step,-predic_step,-1):
                df[index][j].append(series[k - i][j])

    return df

def multi_generate_testdata_by_n_days(series, n, multi,index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = []
    k = 0
    index = -1
    while k + n+predic_step < len(series):
        df.append([])
        index = index + 1
        for j in range(multi):
            df[index].append([])
            for i in range(n):
                df[index][j].append(series[k + i][j])
            for i in range(predic_step,n+predic_step):
                df[index][j].append(series[k + i][j])
        k = k + n
    return df

def readData(df, n=30, all_too=True, index=False):

    df.index = df.index.strftime('%Y-%m-%d %H:%M')
    df_index = df.index.tolist()
    df_column = np.array(df.copy()).tolist()
    i = 0
    l = len(df_column)
    while i < l:
        for j in range(len(df_column[0])):
            if np.isnan(df_column[i][j]) == True:
                del (df_column[i])
                del (df_index[i])
            else:
                i = i + 1
                l = len(df_column)
                if i >= l:
                    break
        l = len(df_column)
    df_column = pd.DataFrame(df_column, index=df_index)
    df_index = df_column.index
    return_index = df_index.copy()
    df_column = df_column.values
    df_numpy_mean = np.mean(df_column,axis=0)
    df_numpy_std = np.std(df_column,axis=0)
    df_column = (df_column - df_numpy_mean) / df_numpy_std
    df_index = list(timefeatures.time_features(df_index.values,1,freq='t'))
    df_column_train = df_column
    df_generate_train = multi_generate_traindata_by_n_days(df_column_train.tolist(), n, multi_features,index=index)
    train_index = generate_data_by_n_days_traindata(df_index,n)
    df_generate_test = multi_generate_testdata_by_n_days(df_column.tolist(),n,multi_features,index=index)
    test_index = generate_data_by_n_days_testdata(df_index,n)

    return df_generate_train, df_generate_test,df_column, train_index,test_index,return_index.tolist(),df_numpy_std,df_numpy_mean


def get_padding_mask(input):

    input1 = np.zeros(input.shape)
    return input1


def get_data(data):
    train_data,test_data, orignal_data, train_data_index,test_data_index,data_index,df_numpy_std,df_numpy_mean = readData(data,n=input_window)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_data_index = np.array(train_data_index)
    test_data_index = np.array(test_data_index)
    print(train_data.shape)
    train_data1 = train_data.copy()
    test_data1 = test_data.copy()
    train_padding = get_padding_mask(train_data1)
    test_padding = get_padding_mask(test_data1)
    train_seq = torch.from_numpy(train_data[:, :,:-input_window])
    train_label = torch.from_numpy(train_data[:, :,-input_window:])
    train_seq_index = torch.from_numpy(train_data_index[:,:-input_window,:])
    train_seq_index = train_seq_index.permute(0, 2, 1)
    train_label_index = torch.from_numpy(train_data_index[:, -input_window:,:])
    
    test_seq = torch.from_numpy(test_data[:, :,:-input_window])
    test_label = torch.from_numpy(test_data[:, :,-input_window:])

    test_seq_index = torch.from_numpy(test_data_index[:,:-input_window,:])
    test_seq_index = test_seq_index.permute(0, 2, 1)
    test_label_index = torch.from_numpy(test_data_index[:, -input_window:,:])

    train_padding = torch.from_numpy(np.array(train_padding[:, :,:-input_window]))
    test_padding = torch.from_numpy(np.array(test_padding[:, :,-input_window:]))


    train_sequence = torch.stack((train_seq, train_label, train_padding), dim=1).type(torch.FloatTensor)
    print("traindata",train_sequence.shape) 
    test_data = torch.stack((test_seq, test_label, test_padding), dim=1).type(torch.FloatTensor)
    return train_sequence.to(device), test_data.to(device),train_seq_index.to(device) ,test_seq_index.to(device),df_numpy_std,df_numpy_mean




def get_batch(source,source_index ,i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    data_index = source_index[i:i + seq_len]
    inputlist = []
    for item in data:
        inputlist.append(item[0])
    input = torch.stack(inputlist, dim=0)
    print("input", input.shape)
    targetlist = []
    for item in data:
        targetlist.append(item[1])
    target = torch.stack(targetlist, dim=0)
    print("target", target.shape)

    paddinglist = []
    for item in data:
        paddinglist.append(item[2])
    padding = torch.stack(paddinglist, dim=0)

    print("padding", padding.shape)
    print("index", data_index.shape)

    return input.to(device), target.to(device), padding.to(device),data_index.to(device)


class ConvLSTM_block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder1 = convlstm.ConvLSTM(input_dim=d_model,
                 hidden_dim=[64, 64, 128],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = torch.unsqueeze(x, dim=1)
        x = self.encoder1(x)
        print("=========ConvLSTM.out",x.shape)
        x = torch.squeeze(x)
        return x

class TCN_block(nn.Module):
    def __init__(self, d_model,kernel_size=2,dropout=0.2):
        super().__init__()
        self.encoder1 = TCN.TemporalConvNet(d_model, [d_model,d_model], kernel_size, dropout=dropout)
    def forward(self, x):
        x = self.encoder1(x.permute(0, 2, 1)).transpose(1, 2)
        return x



class LSTM_block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder1 = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )
        self.encoder2 = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, hidden = self.encoder1(x)
        encode, finalhidden = self.encoder2(x,hidden) 
        return encode

class myAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention=nn.Linear(d_model,d_model)
        self.softmax=nn.Softmax(dim=2)

    def forward(self, queries):
        attn = self.attention(queries)
        scores = self.softmax(attn)
        out = queries+scores
        return out

class nnAR(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention=nn.Linear(d_model,d_model)

    def forward(self, queries):
        attn = self.attention(queries)
        out = queries+attn
        return out

class de_tokenAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention=nn.Linear(d_model,d_model)
        self.softmax=nn.Softmax(dim=2)

    def forward(self, queries):
        attn = self.attention(queries)
        scores = self.softmax(attn)
        out = queries*scores
        out = F.relu(out)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv_1 = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        # self.tokenConv_2 = nn.Conv1d(in_channels=d_model // 2, out_channels=d_model,
        #                              kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv_1(x.permute(0, 2, 1)).transpose(1, 2)
        # x = self.tokenConv_2(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class unsample(nn.Module):
    def __init__(self, c_in, d_model):
        super(unsample, self).__init__()
        self.de_tokenConv_1 = nn.ConvTranspose1d(in_channels=c_in, out_channels=d_model,
                                                 kernel_size=3, padding=1)
        # self.de_tokenConv_2 = nn.ConvTranspose1d(in_channels=c_in // 2, out_channels=d_model,
        #                                          kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.de_tokenConv_1(x.permute(0, 2, 1)).transpose(1, 2)
        # x = self.de_tokenConv_2(x.permute(0, 2, 1)).transpose(1, 2)
        return x
        
class de_TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(de_TokenEmbedding, self).__init__()
        self.de_tokenConv_1 = nn.ConvTranspose1d(in_channels=c_in, out_channels=c_in // 2,
                                   kernel_size=3,padding=1)
        self.de_tokenConv_2 = nn.ConvTranspose1d(in_channels=c_in // 2, out_channels=d_model,
                                     kernel_size=3,padding=1)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.de_tokenConv_1(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.de_tokenConv_2(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        self.freq = freq
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
            self.hour_embed = Embed(hour_size, d_model)
            self.weekday_embed = Embed(weekday_size, d_model)
        if freq=='h':
            self.hour_embed = Embed(hour_size, d_model)
            self.weekday_embed = Embed(weekday_size, d_model)
        if freq=='d':
            self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_flag = False
        hour_flag = False
        if self.freq=='t':
            minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
            hour_x = self.hour_embed(x[:, :, 3])
            weekday_x = self.weekday_embed(x[:, :, 2])
            minute_flag = True
        elif self.freq=='h':
            hour_x = self.hour_embed(x[:, :, 3])
            weekday_x = self.weekday_embed(x[:, :, 2])
            hour_flag = True
        else:
            weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        if minute_flag==False and hour_flag==True:
            minute_x = torch.zeros(weekday_x.shape)
        elif minute_flag==False and hour_flag==False:
            minute_x = torch.zeros(weekday_x.shape)
            hour_x = torch.zeros(weekday_x.shape)
        # return  weekday_x + day_x + month_x
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv_1 = nn.Conv1d(in_channels=5, out_channels=8,
                                     kernel_size=3, padding=padding, padding_mode='circular')

    def forward(self, x,x_mark):
        x = self.position_embedding(x)
        return x




class TransAm_encoder(nn.Module):
    def __init__(self, feature_size=512,num_layers=1, dropout=0.25):
        super(TransAm_encoder, self).__init__()
        self.model_type = 'Transformer'
        self.feature_size = feature_size
        self.predict_features = input_window
        nhead = feature_size//64
        self.src_mask = None
        self.value_embedding_1 = TokenEmbedding(c_in=input_window, d_model=128)
        self.value_embedding_3 = TokenEmbedding(c_in=128, d_model=512)

        self.pos_encoder = DataEmbedding(multi_features,feature_size,'fixed', freq='t')
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=0.4)
        encoder_norm = nn.LayerNorm(feature_size) #LayerNorm
        self.posion_dropout = nn.Dropout(0.1)

        self.transformer_encoder_before = nn.TransformerEncoder(self.encoder_layer, num_layers, encoder_norm)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 1, encoder_norm)

        self.encoder_AR1 = nnAR(feature_size)
        self.encoder_AR2 = nnAR(feature_size)

        self.TCN_encoder = TCN_block(feature_size)

        self.src_key_padding_mask = None 


    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std



    def forward(self, src, tgt,src_padding,data_index):
        inputs = src

        src_2 = self.pos_encoder(src,data_index) 
        tgt_2 = self.pos_encoder(tgt,data_index)

        c1 = src                      
        c2 = self.value_embedding_1(c1)
 
        c4 = self.value_embedding_3(c2)

        src_1 = c4


        c1 = tgt                        
        c2 = self.value_embedding_1(c1)
        c4 = self.value_embedding_3(c2)
        tgt_1 = c4

        src = self.posion_dropout(src_1+src_2)
        tgt = self.posion_dropout(tgt_1+tgt_2)



        print("model_input",src.shape)
        src = self.encoder_AR1(src)
        src = self.TCN_encoder(src) 
        encode_1 = self.transformer_encoder_before(src, self.src_mask, self.src_key_padding_mask) 
        encode_1 = self.encoder_AR2(encode_1)
        encode_1 = self.TCN_encoder(encode_1)#attention
        encode_1 = F.selu(encode_1)

        memory = self.transformer_encoder(encode_1, self.src_mask, self.src_key_padding_mask)
        memory = memory+encode_1+src 
        return memory,tgt


class TransAm_decoder(nn.Module):
    def __init__(self, feature_size=512, num_layers=1,
                 dropout=0.25): 
        super(TransAm_decoder, self).__init__()
        self.model_type = 'Transformer'
        self.feature_size = feature_size
        self.predict_features = input_window
        nhead = feature_size // 64
        self.src_mask = None


        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, dropout=0.4)
        decoder_norm = nn.LayerNorm(feature_size)  # LayerNorm

        self.transformer_decoder_before = nn.TransformerDecoder(decoder_layer, 1, decoder_norm)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        self.de_value_embedding = de_TokenEmbedding(c_in=feature_size, d_model=input_window)
        self.decoder = nn.Linear(input_window, input_window)
        self.dropout = nn.Dropout(dropout)
        self.posion_dropout = nn.Dropout(0.1)

        self.decoder_AR1 = nnAR(feature_size)
        self.decoder_AR2 = nnAR(feature_size)
        self.TCN_decoder = TCN_block(feature_size)

        self.init_weights()
        self.src_key_padding_mask = None  

    def init_weights(self):
        initrange = 0.1  
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, memory, tgt, src_padding, data_index):
        print("memory", memory.shape)
        memory_after = self.transformer_decoder_before(tgt, memory, self.src_mask, self.src_key_padding_mask)
        memory_after = self.decoder_AR1(memory_after)
        memory_after = self.TCN_decoder(memory_after)
        decode_1 = F.selu(memory_after)
        transformer_output = self.transformer_decoder(tgt, decode_1, self.src_mask, self.src_key_padding_mask)
        transformer_output = self.decoder_AR2(transformer_output)
        transformer_output = self.TCN_decoder(transformer_output)
        transformer_output = torch.sigmoid(transformer_output)

        transformer_output = transformer_output + decode_1  

        de_output = self.de_value_embedding(transformer_output)
        de_output = self.dropout(de_output)
        output = self.decoder(de_output)
        output = self.dropout(output)
        return output

#### model stracture ####
class TransAm_discriminator(nn.Module):
    def __init__(self, feature_size=512, num_layers=1,
                 dropout=0.25): 
        super(TransAm_discriminator, self).__init__()
        self.model_type = 'Transformer'
        self.feature_size = feature_size
        self.predict_features = input_window
        nhead = feature_size // 64
        self.src_mask = None

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, dropout=0.4)
        decoder_norm = nn.LayerNorm(feature_size)  

        self.transformer_decoder_before = nn.TransformerDecoder(decoder_layer, 1, decoder_norm)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        self.de_value_embedding = de_TokenEmbedding(c_in=feature_size, d_model=input_window)
        self.decoder = nn.Linear(input_window, 1)
        # self.discriminator = nn.Linear(input_window, 1)
        self.dropout = nn.Dropout(dropout)
        self.posion_dropout = nn.Dropout(0.1)


        self.decoder_AR1 = nnAR(feature_size)
        self.decoder_AR2 = nnAR(feature_size)

        # self.TCN_encoder = TCN_block(feature_size)
        self.TCN_decoder = TCN_block(feature_size)

        self.init_weights()
        self.src_key_padding_mask = None  # 后面用了掩码


    def init_weights(self):
        initrange = 0.1  ？
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, memory, tgt, src_padding, data_index):
        # memory = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        print("memory", memory.shape)
        memory_after = self.transformer_decoder_before(tgt, memory, self.src_mask, self.src_key_padding_mask)
        memory_after = self.decoder_AR1(memory_after)
        memory_after = self.TCN_decoder(memory_after)  # attention
        decode_1 = F.selu(memory_after)
        transformer_output = self.transformer_decoder(tgt, decode_1, self.src_mask, self.src_key_padding_mask)
        # transformer_output = transformer_output+decode_1+memory 
        transformer_output = self.decoder_AR2(transformer_output)
        transformer_output = self.TCN_decoder(transformer_output)  # attention
        transformer_output = torch.sigmoid(transformer_output)

        transformer_output = transformer_output + decode_1  
        # transformer_output = transformer_output+decode_1+memory 

        # transformer_output = self.myAttention(transformer_output)
        # transformer_output = transformer_output*transformer_output_scores
        de_output = self.de_value_embedding(transformer_output)
        # print("de_value_embedding",de_output.shape)
        de_output = self.dropout(de_output)
        output = self.decoder(de_output)
        output = self.dropout(output)
        return F.sigmoid(output)

class TransAm_discriminator2(nn.Module):
    def __init__(self, feature_size=512,num_layers=1, dropout=0.25):
        super(TransAm_discriminator2, self).__init__()
        self.model_type = 'Transformer'
        self.feature_size = feature_size
        self.predict_features = input_window
        nhead = feature_size//64
        self.src_mask = None


        self.value_embedding_1 = TokenEmbedding(c_in=input_window, d_model=128)
        # self.unsample_1 = unsample(c_in=input_window, d_model=64)
        # self.value_embedding_2 = TokenEmbedding(c_in=64, d_model=128)
        # self.unsample_2 = unsample(c_in=64, d_model=128)
        self.value_embedding_3 = TokenEmbedding(c_in=128, d_model=512)
        # self.unsample_3 = unsample(c_in=128, d_model=512)

        self.pos_encoder = DataEmbedding(multi_features,feature_size,'fixed', freq='t')
        # padding = 1 if torch.__version__ >= '1.5.0' else 2
        # self.cnn = nn.Conv1d(in_channels=feature_size, out_channels=feature_size,kernel_size=1, padding=0, padding_mode='circular')
        # self.cnn = nn.Linear(multi_features, feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=0.4)
        encoder_norm = nn.LayerNorm(feature_size) #LayerNorm
        self.posion_dropout = nn.Dropout(0.1)

        self.transformer_encoder_before = nn.TransformerEncoder(self.encoder_layer, num_layers, encoder_norm)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 1, encoder_norm)
        # self.transformer_encoder_mu = nn.TransformerEncoder(self.encoder_layer,1,encoder_norm)
        # self.transformer_encoder_logvar = nn.TransformerEncoder(self.encoder_layer, 1, encoder_norm)
        self.de_value_embedding = de_TokenEmbedding(c_in=feature_size, d_model=input_window)
        # self.decoder_1 = nn.Linear(feature_size, feature_size//2)
        self.decoder = nn.Linear(input_window, 1)
        # self.discriminator = nn.Linear(input_window, 1)
        self.dropout = nn.Dropout(dropout)

        #attention
        # self.ExternalAttention = LSTM_block(feature_size)
        # self.myattention = myAttention(feature_size)
        self.encoder_AR1 = nnAR(feature_size)
        self.encoder_AR2 = nnAR(feature_size)
        # self.decoder_AR1 = nnAR(feature_size)
        # self.decoder_AR2 = nnAR(feature_size)

        self.TCN_encoder = TCN_block(feature_size)
        # self.TCN_decoder = TCN_block(feature_size)
        # self.ExternalAttention = ExternalAttention(feature_size, 64)
        # self.decoder = nn.Linear(feature_size, 1)
        # self.init_weights()
        self.src_key_padding_mask = None 
        # self.src_mask = torch.zeros((160,20))
        # self.src_key_padding_mask = torch.zeros((100,20))

    # def init_weights(self):
    #     initrange = 0.1 
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std



    def forward(self, src, tgt,src_padding,data_index):
        inputs = src
        # print("r_decoder_before(tgt,memory,self.src_mask, scnn_after",src.shape)
        src_2 = self.pos_encoder(src,data_index) 
        tgt_2 = self.pos_encoder(tgt,data_index)
        c1 = src                       
        c2 = self.value_embedding_1(c1)
        c4 = self.value_embedding_3(c2)
        src_1 = c4


        c1 = tgt                        
        c2 = self.value_embedding_1(c1)
        c4 = self.value_embedding_3(c2)
        tgt_1 = c4

        src = self.posion_dropout(src_1+src_2)
        tgt = self.posion_dropout(tgt_1+tgt_2)

        print("model_input",src.shape)
        src = self.encoder_AR1(src)
        src = self.TCN_encoder(src) 
        encode_1 = self.transformer_encoder_before(src, self.src_mask, self.src_key_padding_mask) #
        encode_1 = self.encoder_AR2(encode_1)
        encode_1 = self.TCN_encoder(encode_1)#attention
        encode_1 = F.selu(encode_1)
        memory = self.transformer_encoder(encode_1, self.src_mask, self.src_key_padding_mask)
        memory = memory+encode_1+src 
        de_output = self.de_value_embedding(memory)
        de_output = self.dropout(de_output)
        output = self.decoder(de_output)
        output = self.dropout(output)
        return F.sigmoid(output)




def train(train_data,train_data_index,batch_size,model_encoder,model_decoder,model_discriminator,model_discriminator2):
    model_encoder.train()  # Turn on the train mode
    model_decoder.train()
    model_discriminator.train()
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        print("batch", batch) #i
        print("i",i) #i*batch_size
        if len(train_data) - 1 - i<batch_size:
            break

        data, targets, key_padding_mask,data_index = get_batch(train_data,train_data_index ,i, batch_size)
        optimizer_encoder.zero_grad()
        optimizer_decoder1.zero_grad()
        optimizer_decoder2.zero_grad()
        optimizer_discriminator.zero_grad()
        # reconstruction loss
        memory,tgt = model_encoder(data,targets, key_padding_mask,data_index)
        output = model_decoder(memory,tgt, key_padding_mask,data_index)
        # print("output",output.shape)
        reconst_loss = criterion(output, targets)
        # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # loss = (reconst_loss + kl_div)
        loss = reconst_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model_decoder.parameters(), 0.5)
        optimizer_encoder.step()
        optimizer_decoder1.step()
        # Discriminator
        ## true prior is random normal (randn)
        ## this is constraining the Z-projection to be normal!
        model_encoder.eval()
        z_fake_gauss, tgt = model_encoder(data, targets, key_padding_mask, data_index)
        D_fake_gauss = model_discriminator(z_fake_gauss, tgt, key_padding_mask, data_index)
        z_real_gauss = Variable(torch.randn(data.size()[0], data.size()[1],d_model) * 5.).cuda()
        D_real_gauss = model_discriminator(z_real_gauss,tgt, key_padding_mask,data_index)

        D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
        D_loss.backward()
        optimizer_discriminator.step()
        # Discriminator2
        model_decoder.eval()
        z_fake_gauss, tgt = model_encoder(data, targets, key_padding_mask, data_index)
        x_fake_gauss = model_decoder(z_fake_gauss, tgt, key_padding_mask, data_index)
        D_fake_gauss = model_discriminator2(x_fake_gauss, targets, key_padding_mask, data_index)

        x_real_gauss = data
        D_real_gauss = model_discriminator2(x_real_gauss, targets, key_padding_mask, data_index)
        D2_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
        D2_loss.backward()
        optimizer_discriminator2.step()
        # Generator
        model_encoder.train()
        model_decoder.train()
        z_fake_gauss, tgt = model_encoder(data,targets, key_padding_mask,data_index)
        D_fake_gauss = model_discriminator(z_fake_gauss,tgt, key_padding_mask,data_index)
        G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
        G_loss.backward()
        optimizer_encoder2.step()

        z_fake_gauss, tgt = model_encoder(data, targets, key_padding_mask, data_index)
        x_fake_gauss = model_decoder(z_fake_gauss, tgt, key_padding_mask, data_index)
        D_fake_gauss = model_discriminator2(x_fake_gauss, targets, key_padding_mask, data_index)
        G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
        G_loss.backward()
        optimizer_decoder2.step()


        total_loss += loss.item()
        # print("len(train_data)",len(train_data))
        # print("batch_size",batch_size)
        # print("len(train_data) / batch_size",len(train_data) / batch_size)
        # print("len(train_data) / batch_size")
        log_interval = int(len(train_data) / batch_size)
        # print("log_interval",log_interval)
        # if batch % log_interval == 0 and batch > 0:
        if batch % 10==0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr_encoder {:02.6f} |lr_decoder {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(epoch, batch, len(train_data) // batch_size, scheduler_encoder.get_last_lr()[0],scheduler_decoder1.get_last_lr()[0],   #get_last_lr()
                              elapsed * 1000 / log_interval,
                cur_loss))  # , math.exp(cur_loss)
            total_loss = 0
            start_time = time.time()


def plot_and_loss(model_encoder,model_decoder, data_source,data_source_index ,epoch,batch_size):
    model_encoder.eval()
    model_decoder.eval()
    total_loss = 0.
    eval_batch_size = batch_size
    realY = []
    predictY = []
    realY_anomaly = []
    predictY_anomaly = []
    for i in range(multi_features):
        realY.append([])
        predictY.append([])
        realY_anomaly.append([])
        predictY_anomaly.append([])
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    test_result1 = torch.Tensor(0)
    truth1 = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            if len(data_source) - 1 - i < eval_batch_size:
                break
            data, target, key_padding_mask,data_index = get_batch(data_source, data_source_index,i, eval_batch_size)
            memory,tgt = model_encoder(data, target, key_padding_mask,data_index)
            output = model_decoder(memory, tgt, key_padding_mask,data_index)
            # look like the model returns static values for the output window
            # itemlist = []
            # tgtlist = []
            # outputlist = []
            # for j in range(data.shape[2]):
            #     itemlist.append(data[:, :, j])
            #     tgtlist.append(target[:, :, j])
            # print("itemlist", itemlist[0].shape)
            # for j in range(len(itemlist)):
            #     output_item = eval_model(itemlist[j],tgtlist[j], key_padding_mask)
            #     outputlist.append(output_item)
            #     print("targets", tgtlist[j].shape)
            #     print("output_item", output_item.shape)
            # output = torch.cat(outputlist, dim=2)
            # print("output", output.shape)


            if calculate_loss_over_all_values:
                reconst_loss = criterion(output, target).item()
                # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                # loss = (reconst_loss + kl_div)
                loss = reconst_loss
                total_loss += loss
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            lable_Y = target.cpu().numpy()
            # print("lable_Y", lable_Y.shape)
            for j in range(lable_Y.shape[0]):
                tmp = lable_Y[j]
                for k in range(multi_features):
                    realY[k].extend(tmp[k])
                    realY_anomaly[k].append(tmp[k])


            predic_Y = output.cpu().numpy()
            # print("predic_Y", predic_Y.shape)
            for j in range(predic_Y.shape[0]):
                tmp = predic_Y[j]
                for k in range(multi_features):
                    predictY[k].extend(tmp[k])
                    predictY_anomaly[k].append(tmp[k])


    realY = np.array(realY)
    predictY = np.array(predictY)
    for i in range(multi_features):
        realY[i] = realY[i] * df_numpy_std[i] + df_numpy_mean[i]
        predictY[i] = predictY[i] * df_numpy_std[i] + df_numpy_mean[i]

    pyplot.plot(realY[1], color="red", alpha=0.5, label="real")
    pyplot.plot(predictY[1], color="green", alpha=0.8, label="predic")
    pyplot.savefig('transformer_TCN_1_epo%d.png' % epoch)
    # pyplot.show()
    pyplot.close("all")
    pyplot.plot(realY[0], color="red", alpha=0.5,label="real")
    pyplot.plot(predictY[0], color="green", alpha=0.8,label = "predic")
    pyplot.savefig('transformer_TCN_0_epo%d.png' % epoch)
    # pyplot.show()
    pyplot.close("all")

    return total_loss/i, realY,predictY,realY_anomaly,predictY_anomaly

def eval_model_output(model_encoder,model_decoder, data_source,data_source_index ,batch_size):
    model_encoder.eval()
    model_decoder.eval()
    total_loss = 0.
    eval_batch_size = batch_size
    realY_anomaly = []
    predictY_anomaly = []
    realY = []
    predictY = []
    for i in range(multi_features):
        realY.append([])
        predictY.append([])
        realY_anomaly.append([])
        predictY_anomaly.append([])
    # print("len(data_source)",len(data_source))
    # print("eval_batch_size",eval_batch_size)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            if len(data_source) - 1 - i < eval_batch_size:
                break
            data, target, key_padding_mask,data_index = get_batch(data_source, data_source_index,i, eval_batch_size)
            memory,tgt = model_encoder(data, target, key_padding_mask,data_index)
            output = model_decoder(memory,tgt, key_padding_mask,data_index)
            if calculate_loss_over_all_values:
                reconst_loss = criterion(output, target).item()
                # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                # 反向传播及优化器
                # loss = (reconst_loss + kl_div)
                loss = reconst_loss
                total_loss += loss
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            lable_Y = target.cpu().numpy()
            for j in range(lable_Y.shape[0]):
                tmp = lable_Y[j]
                for k in range(multi_features):
                    realY[k].extend(tmp[k])
                    realY_anomaly[k].append(tmp[k])

            predic_Y = output.cpu().numpy()
            # print("predic_Y", predic_Y.shape)
            for j in range(predic_Y.shape[0]):
                tmp = predic_Y[j]
                for k in range(multi_features):
                    predictY[k].extend(tmp[k])
                    predictY_anomaly[k].append(tmp[k])

    realY = np.array(realY)
    predictY = np.array(predictY)
    realY_anomaly = np.array(realY_anomaly)
    predictY_anomaly = np.array(predictY_anomaly)
    # print("realY",realY.shape)
    # print("predictY",predictY.shape)
    # print("df_numpy_std",df_numpy_std.shape)
    # print("df_numpy_mean",df_numpy_mean.shape)
    for i in range(multi_features):
        realY[i] = realY[i] * df_numpy_std[i] + df_numpy_mean[i]
        predictY[i] = predictY[i] * df_numpy_std[i] + df_numpy_mean[i]
        for j in range(realY_anomaly.shape[1]):
            realY_anomaly[i][j] = realY_anomaly[i][j]* df_numpy_std[i] + df_numpy_mean[i]
            predictY_anomaly[i][j] = predictY_anomaly[i][j]* df_numpy_std[i] + df_numpy_mean[i]

    # pyplot.plot(realY[1], color="red", alpha=0.5,label="real")
    # pyplot.plot(predictY[1], color="green", alpha=0.8,label = "predic")
    # pyplot.savefig('mytransformer_useSKAB.png' )
    # pyplot.plot.legend()
    # pyplot.show()
    # pyplot.close("all")

    return realY,predictY,realY_anomaly,predictY_anomaly

def evaluate(model_encoder,model_decoder, data_source,data_source_index,batch_size):
    model_encoder.eval()  # Turn on the evaluation mode
    model_decoder.eval()
    total_loss = 0.
    eval_batch_size = batch_size
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            if len(data_source) - 1 - i < eval_batch_size:
                break
            data, targets, key_padding_mask,data_index = get_batch(data_source,data_source_index ,i, eval_batch_size)
            memory,tgt = model_encoder(data, targets,key_padding_mask,data_index)
            output = model_decoder(memory, tgt,key_padding_mask,data_index)
            if calculate_loss_over_all_values:
                reconst_loss = criterion(output, targets).cpu().item()
                # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                # loss = (reconst_loss + kl_div)
                loss = reconst_loss
                total_loss += len(data[0]) * loss
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)

import sys
sys.path.append('../SKAB_data/utils')
from evaluating import evaluating_change_point
# benchmark files checking
all_files=[]
import os
for root, dirs, files in os.walk("../SKAB_data/data/"):
    for file in files:
        if file.endswith(".csv"):
             all_files.append(os.path.join(root, file))
anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0], sep=';', index_col='datetime', parse_dates=True)
columns = anomaly_free_df.columns
multi_features = len(columns)

train_data, val_data,train_data_index ,val_data_index,df_numpy_std,df_numpy_mean = get_data(anomaly_free_df)

model_encoder = TransAm_encoder().to(device)
model_decoder = TransAm_decoder().to(device)
model_discriminator = TransAm_discriminator().to(device)
model_discriminator2 = TransAm_discriminator2().to(device)

criterion = nn.MSELoss()
# lr = 0.00001
lr = 0.0002
lr_reg = 0.00005
EPS = 1e-15
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer_encoder = torch.optim.AdamW(model_encoder.parameters(), lr=lr)
optimizer_encoder2 = torch.optim.AdamW(model_encoder.parameters(), lr=lr_reg)
optimizer_decoder1 = torch.optim.AdamW(model_decoder.parameters(), lr=lr)
optimizer_decoder2 = torch.optim.AdamW(model_decoder.parameters(), lr=lr_reg)
optimizer_discriminator = torch.optim.AdamW(model_discriminator.parameters(), lr=lr_reg)
optimizer_discriminator2 = torch.optim.AdamW(model_discriminator2.parameters(), lr=lr_reg)

scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, 3, gamma=0.98)
scheduler_encoder2 = torch.optim.lr_scheduler.StepLR(optimizer_encoder2, 3, gamma=0.98)
scheduler_decoder1 = torch.optim.lr_scheduler.StepLR(optimizer_decoder1, 3, gamma=0.98)
scheduler_decoder2 = torch.optim.lr_scheduler.StepLR(optimizer_decoder2, 3, gamma=0.98)
scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optimizer_discriminator, 3, gamma=0.98)
scheduler_discriminator2 = torch.optim.lr_scheduler.StepLR(optimizer_discriminator2, 3, gamma=0.98)

best_val_loss = float("inf")
# epochs = 200  # The number of epochs
epochs = 200
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data,train_data_index,batch_size,model_encoder,model_decoder,model_discriminator,model_discriminator2)
    #train_loss = evaluate(model, train_data)
    # val_loss, tran_output, tran_true, tran_output5, tran_true5 = plot_and_loss(model, val_data, val_data_index, epoch)
    if (epoch % 5 == 0):
        val_loss, tran_output, tran_true, tran_output5, tran_true5 = plot_and_loss(model_encoder,model_decoder, val_data,val_data_index,epoch,30)
        # predict_future(model, val_data, 200)
    else:
        val_loss = evaluate(model_encoder,model_decoder, val_data,val_data_index,30)

    print('-' * 89)
    # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | train loss {:5.5f} '.format(epoch, (time.time() - epoch_start_time),val_loss, train_loss))  # , math.exp(val_loss) | valid ppl {:8.2f}
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f}'.format(epoch, (
                time.time() - epoch_start_time), val_loss))
    print('-' * 89)
    scheduler_encoder.step()
    scheduler_encoder2.step()
    scheduler_decoder1.step()
    scheduler_decoder2.step()
    scheduler_discriminator.step()
    scheduler_discriminator2.step()



PATH = './transformer_TCN_AAE_encoder_GAN_layerchange.pth'
state = {'model': model_encoder.state_dict(), 'optimizer': optimizer_encoder.state_dict(),'optimizer2': optimizer_encoder2.state_dict()}
torch.save(state, PATH)
PATH = './transformer_TCN_AAE_decoder_GAN_layerchange.pth'
state = {'model': model_decoder.state_dict(), 'optimizer': optimizer_decoder1.state_dict(),'optimizer2': optimizer_decoder2.state_dict()}
torch.save(state, PATH)
PATH = './transformer_TCN_AAE_discriminator_GAN_layerchange.pth'
state = {'model': model_discriminator.state_dict(), 'optimizer': optimizer_discriminator.state_dict()}
torch.save(state, PATH)
PATH = './transformer_TCN_AAE_discriminator2_GAN_layerchange.pth'
state = {'model': model_discriminator2.state_dict(), 'optimizer': optimizer_discriminator2.state_dict()}
torch.save(state, PATH)
print("modelfinish")
