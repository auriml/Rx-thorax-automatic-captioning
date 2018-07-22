import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import os
import re
import hashlib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from sklearn.metrics import f1_score, accuracy_score
from sklearn.externals import joblib


from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
import argparse
import remotedebugger as rd
parser = argparse.ArgumentParser()
rd.attachDebugger(parser)

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)


de = 100 # dimension word embedding

def string2numeric_hash(text):
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)

def load_data(de):

    train = []
    labels = []
    texts = []
    seq_lengths = []
    sent_labels = '/manual_review/labeled_sent_28K.csv'
    path = root + '/Rx-thorax-automatic-captioning' + sent_labels
    column_names = ['text','topic', 'counts']
    column_names.extend(list('123456789'))
    sent_labels = pd.read_csv(path, sep = ',',  header = 0, names =column_names)
    def labels2set(x):
        pattern = re.compile('[^a-zA-Z,\s\-]+')
        s = set(pd.Series([pattern.sub('', item.strip()) for item in x.values if not pd.isna(item)]))
        return s
    sent_labels['labels_set'] = sent_labels[list('123456789')].apply(lambda x: labels2set(x), axis=1)

    mlb = MultiLabelBinarizer()
    fitted_mlb = mlb.fit(sent_labels['labels_set'].values)
    joblib.dump(mlb, 'mlb.pkl')


    from fastText import load_model

    path = root + '/Rx-thorax-automatic-captioning' + '/embeddings/fasttext/text.bin'
    f = load_model(path)
    words, frequency = f.get_words(include_freq=True)


    for idx,row in sent_labels.iterrows():
        s = sent_labels.ix[idx,'text']
        l = sent_labels.ix[idx,'labels_set']
        if not pd.isna(s):
            train.append([f.get_word_vector(w) for w in str(s).split(' ') ])
            texts.append([s,l])
            t = fitted_mlb.transform(np.array([l]))
            labels.append(t[0])
            #print(l)
            #print(fitted_mlb.inverse_transform(t))


    

    
    b = np.zeros([len(train),len(max(train,key = lambda x: len(x))), de])
    for i,j in enumerate(train):
        if len(j) > 0:
            b[i][0:len(j)] = np.array(j)
            seq_lengths.append(len(j))
        else:
            b[i][0:len(j)] = np.zeros([1,de])
            seq_lengths.append(0)

    np.save('seq_lengths',seq_lengths)
    np.save('train',b)
    np.save('text', texts)
    np.save('labels',labels)


    #labels = [pattern.sub('', item.strip()) for sublist in labels for item in sublist if not pd.isna(item)]
    #labels = set(labels)
    #idx2labels = {string2numeric_hash(l): l for l in labels}

    return train, labels, texts
#load_data(de)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sent_Dataset(Dataset):
    def __init__(self, transforms=None, CNN_format=False):
        # stuff
        ...
        self.transforms = transforms
        data = np.load('train.npy')
        if CNN_format:
            data = np.transpose(data, [0,2,1])
        else:
            pass
        self.seq_lenghts_array = np.load('seq_lengths.npy')
        self.data_array = data
        self.labels_array = np.load('labels.npy')
        self.texts_array = np.load('text.npy')
        self.mlb = joblib.load('mlb.pkl')

    def __getitem__(self, index):

        data =self.data_array[index]
        data = torch.FloatTensor(data, device = device)
        label = self.labels_array[index]
        label = torch.FloatTensor(label, device = device).view(1, -1)
        length = self.seq_lenghts_array[index]
        

        if self.transforms is not None:
            data = self.transforms(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return ((index, data, length), label)

    def __len__(self):
        return len(self.data_array)

class BatchSamplerOrdered(BatchSampler):
    def __iter__(self):
            batch = []
            lengths = []
            for idx in self.sampler:
                batch.append(int(idx))
                lengths.append(sent_dataset.seq_lenghts_array[idx])
                if len(batch) == self.batch_size:
                    merge = list(zip(batch,lengths))
                    merge.sort(key=lambda tup: tup[1])
                    batch, _ = zip(*merge[::-1])
                    yield list(batch)
                    #yield batch
                    batch = []
                    lengths = []
            if len(batch) > 0 and not self.drop_last:
                merge = list(zip(batch,lengths))
                merge.sort(key=lambda tup: tup[1])
                batch, _ = zip(*merge[::-1])
                yield list(batch)
                #yield batch

batch_size = 1024
sent_dataset =  Sent_Dataset(CNN_format=False)
idx_train, idx_valid = train_test_split(range(sent_dataset.__len__()), test_size=0.1, random_state=42)
train_sampler = BatchSamplerOrdered(SubsetRandomSampler(idx_train[:]), batch_size, False)
valid_sampler = BatchSamplerOrdered(SubsetRandomSampler(idx_valid[:]), batch_size, False)

train_loader = DataLoader(sent_dataset, batch_sampler=train_sampler, shuffle=False, num_workers=4,
                          pin_memory=True if torch.cuda.is_available() else False)
valid_loader = DataLoader(sent_dataset, batch_sampler=valid_sampler, shuffle=False,num_workers=4,
                          pin_memory=True if torch.cuda.is_available() else False)


class _classifier(nn.Module):
    def __init__(self, nlabel, de):
        super(_classifier, self).__init__()
        self.conv1 = nn.Conv1d(de, 64, 3, padding=2)
        self.fc1 = nn.Linear(64*58, nlabel)

    def forward(self, input):
        input = input[1]
        x = F.relu(self.conv1(input))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


class _classifierCNN(nn.Module):
    def __init__(self, nlabel, de):
        super(_classifierCNN, self).__init__()
        self.conv1 = nn.Conv1d(de, 64, 3, padding=2)
        self.pooling = nn.MaxPool1d(kernel_size = 2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=2)
        self.fc1 = nn.Linear(128*31, nlabel)

    def forward(self, input):
        input = input[1]
        x = F.relu(self.conv1(input))
        x = self.pooling(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*1, num_classes)  # 2 for bidirection
    
    def forward(self, input):
        # Set initial states
        x = input[1]

        if np.random.rand() < 0.00001:
            print(x)
        #sequences should be sorted by length in a decreasing order, i.e. input[:,0] should be the longest sequence, and input[:,B-1] the shortest one.
        lengths = input[2]
        x = nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True)
        #h0 = torch.zeros(self.num_layers*1, lengths[0], self.hidden_size).to(device) # 2 for bidirection 
        #c0 = torch.zeros(self.num_layers*1, lengths[0], self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out,  (ht, ct) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        #out, out_length = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        #print(out.size())
        #out = out[:,out_length-1,:]#[out[i, length-1, :] for i, length in enumerate(lengths)]
        #print(out.size())
        #out = out.view(out.shape[0], -1)
        #print(out.size())
        
        #out = self.fc(out)
        out = ht[-1]
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)   

        return out


class MultilabelCategoricalAccuracy(Metric):
    """
    Calculates the multilabel categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
    - `y` must be in the following shape (batch_size, ...)
    """
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        self._macro_accuracies = []
        self._micro_accuracies = []
        self._accuracies = []


    def update(self, output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred).data > 0.5
        y = y.view(y.shape[0],-1)


        self._macro_accuracies.append(f1_score(y, y_pred, average='macro'))
        self._micro_accuracies.append(f1_score(y, y_pred, average='micro'))
        self._micro_accuracies.append(f1_score(y, y_pred, average='weighted'))
        self._accuracies.append(accuracy_score(y, y_pred))


    def compute(self):
        #if self._num_examples == 0:
            #raise NotComputableError('CategoricalAccuracy must have at least one example before it can be computed')

        return np.average(self._accuracies),np.average(self._macro_accuracies),np.average(self._micro_accuracies)


nlabel = len(sent_dataset.labels_array[0])
#classifier = _classifierCNN(nlabel,de)
num_layers = 2
hidden_size = 128
input_size = de
classifier = BiRNN(input_size, hidden_size, num_layers, nlabel).to(device)

optimizer = optim.RMSprop(classifier.parameters())
criterion = nn.MultiLabelSoftMarginLoss()


trainer = create_supervised_trainer(classifier, optimizer,criterion, device=device)

evaluator = create_supervised_evaluator(classifier, metrics= {'accuracy': MultilabelCategoricalAccuracy(),
                                                              'nll': Loss(criterion)}, device=device)



print(device)
log_interval = 10
log_view_interval = 10
epochs = 100
validate_every = 100
checkpoint_every = 100
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % log_interval == 0:
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.7f}"
              "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_view(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % log_view_interval == 0:
        #for idx in engine.state.batch[0][0]:
        #    print(sent_dataset.texts_array[idx])
        #    print(sent_dataset.mlb.inverse_transform(np.array([sent_dataset.labels_array[idx]])))
        y_pred = torch.sigmoid(engine.state.y_pred).data > 0.5
        y = engine.state.y.view(engine.state.y.shape[0],-1)
        yl_pred = sent_dataset.mlb.inverse_transform(y_pred.cpu().detach().numpy())
        yl = sent_dataset.mlb.inverse_transform(y.cpu().detach().numpy())
        z = list(zip(yl,yl_pred))
        #print("ground_truth,predicted")
        #print(z)

            #print("Epoch[{}] Iteration[{}/{}] Loss: {:.7f}"
            #  "".format(engine.state.epoch, iter, len(train_loader.batch_sampler), engine.state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    print("Training Results - Epoch: {}  Avg accuracy: {} Avg loss: {:.7f}"
          .format(engine.state.epoch, avg_accuracy, avg_nll))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(valid_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    print("Validation Results - Epoch: {}  Avg accuracy: {} Avg loss: {:.7f}"
          .format(engine.state.epoch, avg_accuracy, avg_nll))

#@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_view(engine):
    evaluator.run(valid_loader)
    y_pred, y = evaluator.state.output
    y_pred = torch.sigmoid(y_pred).data > 0.5
    y = y.view(y.shape[0],-1)
    yl_pred = sent_dataset.mlb.inverse_transform(y_pred.cpu().detach().numpy())
    yl = sent_dataset.mlb.inverse_transform(y.cpu().detach().numpy())
    yt = np.take(sent_dataset.texts_array[:,0],evaluator.state.batch[0][0])

    z = list(zip(yt,yl,yl_pred))
    print("VALIDATION: text, ground_truth, predicted")
    for i in z:
        print(i)

@trainer.on(Events.EPOCH_COMPLETED)       
def save_model(trainer, model):
     out_path = "model_epoch_{}.pth".format(trainer.current_epoch)
     torch.save(model.state_dict, out_path)

#the_model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))
#testor.add_event_handler(Events.COMPLETED, get_performance(logger)) model.load_state_dict(torch.load('../model/' + model_name))
#testor.run(test_loader)

trainer.run(train_loader, max_epochs=epochs)


