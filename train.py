from torchvision import transforms
from dataset import Flickr8K
from torch.utils.data import DataLoader
import torch
from models import ImageCaptioner
import torch.nn as nn
import torch.optim as optimizers
from caption import caption_image


transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

train = Flickr8K('dataset/Images','dataset/captions.txt',transform)

trainloader = DataLoader(train,batch_size=64)

vocab_size = len(train.vocab.wti)
embed_dim = 200
hidden_size = 512
pad_idx = train.vocab.wti.get('<PAD>')
lr = 0.002

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ImageCaptioner(vocab_size,embed_dim,hidden_size,pad_idx).to(device)

EPOCHS = 50
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optimizers.Adam(model.parameters(),lr)

for epoch in range(EPOCHS):

  for images,inputs,targets in trainloader:

    batch_size = inputs.shape[0]
    hidden = torch.zeros(1, batch_size, hidden_size).to(device)
    cell = torch.zeros(1, batch_size, hidden_size).to(device)
    images = images.to(device)
    inputs = inputs.to(device) 
    targets = targets.to(device)

    outputs,(_,_) = model(images,inputs,hidden,cell)

    loss = criterion(outputs.reshape(outputs.shape[0]*outputs.shape[1],-1),targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  print(f'EPOCH:{epoch},LOSS:{loss}')
  if(EPOCHS % 10 == 0):
    checkpoint = {
        'model':model,
        'model_state_dict':model.state_dict(),
        'vocabulary':train.vocab.wti
        }
    torch.save(checkpoint,'Pretrained/ImageCaptioner.pth')
