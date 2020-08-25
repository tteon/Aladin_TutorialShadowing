import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint


spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

# 'Hello my name is' -> ['Hello', 'my', 'name', 'is']

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True,
               init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_eng, lower=True,
               init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding_size = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape : (seq_length, N)

        embedding = self.dropout(self.embedding_size(x))
        # embedding shape ; (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):

        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size) # mapping
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    ### focus on how tweaking these parameters of shape ( squeeze , unsqueeze )

    def forward(self, x, hidden, cell):
        # shape of x ; (N) but we want (1, N) # one word at a time ; prediction task
        x = x.unsqueeze(0) # add one dimension

        embedding = self.dropout(self.embedding(x))
        # embedding shape; (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs ; (1, N, hidden_size)

        predictions = self.fc(outputs)
        # shape of predictions ; (1, N, length_of_vocab)

        predictions = predictions.squeeze(0) # remove above dimension 1 -> 1

        return  predictions, hidden, cell


# 10000
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        # (target_len, N)
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden , cell = self.encoder(source)

        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden,  cell)

            outputs[t] = output

            # output will be (N, english_vocab_size)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess # random.random means int which is between 0 and 1

        return outputs

### Now we're ready to do the training ###

# Training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64


# Model hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator , test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src), # vary in length , it's gonna prioritze to have examples that are of similar them , padding , save on compute
    device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # we don't want to pay anything about that

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'),model, optimizer)

sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."


for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')

    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )
    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # output shape ; (tar_len, batch_size, output_dim)
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it

        # (N, 10) and targets would be (N)
        output = output[1:].reshape(-1, output.shape[2]) # output dimension which would be the size of the vocabulary and just gonna put everyting else together
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # To avoid exploding problems deprecated errors happen
        #clipping_value = 1  # arbitrary value of your choosing
        #torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        clipping_value = 1  # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        # Gradient descent step
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1

score = bleu(test_data, model, german, english, device)
print(f'Bleu score {score*100:.2f}')