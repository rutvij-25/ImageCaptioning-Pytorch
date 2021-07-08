import torch.nn as nn
import torch
import torchvision

class CNNEncoder(nn.Module):

  def __init__(self,embed_size):
    super(CNNEncoder,self).__init__()
    self.CNN = torchvision.models.vgg16(pretrained=True)
    for params in self.CNN.parameters():
      params.requires_grad = False
    self.Linear = nn.Linear(25088,embed_size)

  def forward(self,x):
    x = self.CNN.features(x)
    x = x.reshape((x.shape[0],-1))
    x = self.Linear(x)
    return x


class RNNDecoder(nn.Module):

    def __init__(self,vocab_size,embed_dim,hidden_size,num_layers = 1):

        super(RNNDecoder,self).__init__()
        self.LSTM = nn.LSTM(embed_dim,hidden_size,num_layers,batch_first=True)
        self.Linear = nn.Linear(hidden_size,vocab_size)
    
    def forward(self,x,h,c):
      
        outputs,(hidden,cell) = self.LSTM(x,(h,c))
        o = self.Linear(outputs)
        return o,(hidden,cell)


class ImageCaptioner(nn.Module):

  def __init__(self,vocab_size,embed_dim,hidden_size,pad_idx,num_layers = 1):
    
    super(ImageCaptioner,self).__init__()
    self.cnn = CNNEncoder(embed_dim)
    self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
    self.rnn = RNNDecoder(vocab_size,embed_dim,hidden_size,num_layers)

  def forward(self,images,captions,h,c):

    features = self.cnn(images).unsqueeze(1)
    embeds = self.embedding(captions)
    inputs = torch.cat([features,embeds],dim=1)
    outputs = self.rnn(inputs,h,c)
    return outputs