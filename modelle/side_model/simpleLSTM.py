# Sentiment analysis of movie reviews

# std libraries
import os
import sys
import time
import re 

# other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contractions

# everything pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# everything torchtext
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe
from torchtext.data import TabularDataset, Iterator, BucketIterator

# everything spacy
import spacy
from spacy.lemmatizer import Lemmatizer
spacy_en = spacy.load('en_core_web_sm')

############################################
#####   Define and check Input File PATH 
############################################
path = '../input'
print(os.listdir(f'{path}'))

torch.manual_seed(1)

############################################
#####   Function Definitions
############################################

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_length, hidden_size, output_size, num_layers=1, bias=True, dropout=0, weights=None):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_length)

        # If weights are given to constructor assign the look-up table to pre-trained word embedding.
        if(weights): self.embedding.weight = nn.Parameter(weights, requires_grad=False)

        self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers=num_layers, bias=bias, dropout=dropout)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        features = self.embedding(x)
        # since h_0 and c_0 are not provided they default to 0
        output, (hidden, cell) = self.lstm(features)
        return self.dense(hidden[-1])

def get_labels(predictions):
    max_idx = torch.max(predictions,1)[1]
    return max_idx
    
# return accuracy per batch
def accuracy(pred_labels, y):
    correct = (pred_labels == y).float() # convert into float for division 
    acc = correct.sum()/len(correct)
    return acc

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train(model, iterator, optimizer, loss_fn):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        prediction = model(batch.text).squeeze(1)
        loss = loss_fn(prediction, batch.label.long())

        prediction_labels = get_labels(prediction).float()
        acc = accuracy(prediction_labels, batch.label)

        loss.backward()
        #clip_gradient(model, 1e-1)
        optimizer.step()

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss/len(iterator), total_epoch_acc/len(iterator)


def eval_model(model, iterator, loss_fn):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:

            prediction = model(batch.text).squeeze(1)
            loss = loss_fn(prediction, batch.label.long())

            pred_labels = get_labels(prediction).float()
            acc = accuracy(pred_labels, batch.label) 
            
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)


def clean(text):
    text = re.sub( '\s+', ' ',  text).strip()  # remove duplicate whitespaces   
    text = contractions.fix(text)  # replace contractions
    return text

# Extract the lemma for each token and join
def lemmatize(text):
    doc = spacy_en(text)
    return " ".join([token.lemma_ for token in doc])

def tokenizer(text): # create a tokenizer function 
    text = clean(text) 
    # text = lemmatize(text)
    tokens = [tok.text for tok in spacy_en.tokenizer(text)]
    if (len(tokens) == 0):
        tokens = ['.']
    return tokens

############################################
#####   Script Parameters
############################################

learning_rate = 2e-5
batch_size = 32
epochs = 4

output_size = 5
hidden_size = 256
embedding_length = 300
bias = True
num_layers = 2
dropout = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################
#####   Data Loading
############################################

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True,
                    include_lengths=True, batch_first=True, fix_length=200)
LABEL = data.LabelField(dtype=torch.float64)

train_data = TabularDataset(f'{path}/train.tsv', 'tsv', fields=[('PhraseId', None), ('SentenceId', None), ('text', TEXT), ('label', LABEL)], skip_header=True)
test_data = TabularDataset(f'{path}/test.tsv', 'tsv', fields=[('PhraseId', None), ('SentenceId', None), ('text', TEXT)], skip_header=True)

TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train_data)

print(f"Length of Text Vocabulary: {len(TEXT.vocab)}")
print(f"Vector size of Text Vocabulary: {TEXT.vocab.vectors.size()}")
print(f"Label Length: {len(LABEL.vocab)}")

train_data, validate_data = train_data.split(split_ratio=0.8)
 
val_iter = BucketIterator(
    validate_data,
    batch_size=batch_size,
    device=device,
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    shuffle = True,
    repeat=False
)

train_iter = BucketIterator(
    train_data, 
    batch_size=batch_size, 
    device=device, 
    sort_key=lambda x: len(x.text), 
    sort_within_batch=False,
    shuffle = True, 
    repeat=False 
)

test_iter = Iterator(
    test_data, 
    batch_size=batch_size, 
    sort=False, 
    device=device, 
    sort_within_batch=False, 
    repeat=False
)

############################################
#####   Model
############################################

emb_weights = TEXT.vocab.vectors
vocab_size = len(TEXT.vocab)

model = LSTM(vocab_size, embedding_length, hidden_size, output_size, num_layers, bias, dropout, weights=None)# weights=emb_weights)

optimizer = optim.Adam(model.parameters())

loss_fn = nn.CrossEntropyLoss()

#set device
model = model.to(device)
loss_fn = loss_fn.to(device)

loss = np.empty([2, epochs])
acc = np.empty([2, epochs])

for epoch in range(epochs):
    train_loss, train_acc = train(model, train_iter, optimizer, loss_fn)
    val_loss, val_acc = eval_model(model, val_iter, loss_fn)

    loss[0, epoch] = train_loss
    loss[1, epoch] = val_loss
    acc[0, epoch] = train_acc
    acc[1, epoch] = val_acc
    
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}% |')

############################################
#####   Prediction
############################################

#TODO predict test data

def save_plot(train, validate, ylabel, plotname, N_EPOCHS):
    epochs = np.arange(N_EPOCHS)+1
    plt.plot(epochs,train, 'b', label='train')
    plt.plot(epochs,validate, 'g', label='validate')
    
    plt.xlabel('epochs')
    plt.ylabel(ylabel)
    plt.legend(loc='upper right', shadow=False)
    plt.savefig(plotname)
    plt.close()

save_plot(loss[0,:], loss[1,:], 'loss', 'LSTM_loss.png', epochs)
save_plot(acc[0,:], acc[1,:], 'accuracy','LSTM_acc.png', epochs)

def predict(model):
    prediction = torch.zeros(len(test_data))
    with torch.no_grad():
        for i in range(len(test_data)):
            test_sen = test_data[i].text
            test_sen = [TEXT.vocab.stoi[x] for x in test_sen]
            tensor = torch.LongTensor(test_sen).to(device)
            tensor = tensor.unsqueeze(1) # tensor has shape [len(test_sen) x 1]
            output = model(tensor)
            out = nn.functional.softmax(output, 1)
            pred_idx = torch.argmax(out[0]) 
            prediction[i] = int(LABEL.vocab.itos[pred_idx])
    return prediction

sub = pd.read_csv(f'{path}/sampleSubmission.csv')

sub.Sentiment = predict(model)
sub.Sentiment = sub.Sentiment.astype(int)
sub.to_csv('submission_Lstm.csv', header = True, index=False)