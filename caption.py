from models import ImageCaptioner
from PIL import Image
import torch
import torchvision.transforms as transform
import torch.nn as nn
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--hidden_size',type=int,default=512)
argparser.add_argument('--n',type=int,default=25)

args = argparser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saved = torch.load('Pretrained/ImageCaptioner.pth',map_location=device)
vocab = saved['vocabulary']
model = saved['model']
model.load_state_dict(saved['model_state_dict'])


def caption_image(model,n,root,vocab,hidden_size):
  image = Image.open(root).convert('RGB')
  transformed_image = transform(image).unsqueeze(0)
  hidden = torch.zeros(1, 1, hidden_size)
  cell = torch.zeros(1, 1, hidden_size)
  with torch.no_grad():
    model.to('cpu')
    features = model.cnn(transformed_image).unsqueeze(0)
    output_s = []
    model.eval()
    softmax = nn.Softmax(dim=0)
    for i in range(n):
      outputs,(hidden,cell) = model.rnn(features,hidden,cell)
      output = torch.argmax(softmax(outputs.squeeze(0).squeeze(0))).item()
      output_s.append(output)
      features = torch.LongTensor([output]).unsqueeze(0)
      features = model.embedding(features)
      if(output == vocab.wti.get('<EOS>')):
        break
  model.to('cuda')
  model.train()  
  return " ".join(vocab.to_text(output_s))


print(caption_image(model,args.n,vocab,args.hidden_size))

