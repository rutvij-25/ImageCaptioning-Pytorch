from PIL import Image
import torch
import torchvision.transforms as transform
import torch.nn as nn


def caption_image(model,n,root,data,hidden_size):
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
      if(output == data.vocab.wti.get('<EOS>')):
        break
  model.to('cuda')
  model.train()  
  return " ".join(data.vocab.to_text(output_s))



