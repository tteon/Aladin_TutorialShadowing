# rooms for improvement
## tweak type of parameters
### create a test set and a sort of a separate validation set
#### tweaking hyperparameters

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)), # CNN takes input 299 x 299
            transforms.ToTensor(),
            transforms.Normalize((0.5 , 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder = 'flickr8k/images',
        annotation_file = 'flickr8k/captions.txt',
        transform = transform,
        num_workers = 2,
    )

    # model configuration
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False
    save_model = False
    train_CNN = False

    # Hyperparameters
    ## We can increase capacity
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    laerning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter('runs/flickr')
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer) # we're returning step here so that the loss fucntions continues where it ended

    model.train()

    for epoch in range(num_epochs):
        print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "step":step,

            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1]) # we actually learn to predict the end token so we're not going to send in the end token
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)) #predicting for each example we're predicting for a bunch of different time steps
            # example , 20 words that it's predicting and then each word has its logit corresponding to each word in the vocabulary right here.
            ## so we have three dimensions here , but the criterion only 2 dimension
            ### output -> (seq_len, N, vocabulary_size) , target -> (seq_len , N)

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
'''
visualization version during training 
for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

'''

if __name__ == "__main__":
    train()


