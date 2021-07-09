from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd

class Vocabulary:
    
    def buildvocab(self,text):
        ## inputs list of sentences -> ['i am iron man','i can do this all day']
        self.vocab = set([j.lower() for i in text for j in i.split(' ')] + ['<SOS>','<EOS>','<UNK>','<PAD>'])
        self.wti = {j:i for (i,j) in enumerate(self.vocab)}
        self.itw = {i:j for (i,j) in enumerate(self.vocab)}
    
    def tokenize(self,text):
        text = ['<SOS>']+text.split(' ')+['<EOS>']
        return [self.wti.get(i,self.wti['<UNK>']) for i in text]
    
    def pad(self,maxlen,tokens):
        return tokens + [self.wti['<PAD>']]*(maxlen - len(tokens))
      
    def to_text(self,tokens):
      return [self.itw.get(i,self.itw.get('<UNK>')) for i in tokens]


class Flickr8K(Dataset):
    
    def __init__(self,root,ann_file,transform = None):
        self.root = root
        self.ann_file = ann_file
        self.transform = transform
        self.vocab = Vocabulary()
        self.data = pd.read_csv(self.ann_file)
        self.img = self.data['image']
        self.captions = self.data['caption'].tolist()
        self.vocab.buildvocab(self.captions)
        
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self,idx):
        cap = self.vocab.tokenize(self.captions[idx])
        inputs = cap[:-1]
        targets = cap[:]
        inputs = self.vocab.pad(39,inputs)
        targets = self.vocab.pad(40,targets)
        im = Image.open(self.root + '/' + self.img[idx]).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im,torch.LongTensor(inputs),torch.LongTensor(targets)
        
