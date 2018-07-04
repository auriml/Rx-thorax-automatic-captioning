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


from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)



def string2numeric_hash(text):
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)

def load_data():
    train = []
    labels = []
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
    labels = np.array(mlb.fit_transform(sent_labels['labels_set'].values))
    print(mlb.classes_)
    np.save('labels',labels)

    from fastText import load_model

    path = root + '/Rx-thorax-automatic-captioning' + '/embeddings/fasttext/text.bin'
    f = load_model(path)
    words, frequency = f.get_words(include_freq=True)

    for s in sent_labels['text'].values:
        train.append([f.get_word_vector(w) for w in str(s).split(' ') if not pd.isna(s)  ])

    de = 100 # dimension word embedding
    b = np.zeros([len(train),len(max(train,key = lambda x: len(x))), de])
    for i,j in enumerate(train):
        if len(j) > 0:
            b[i][0:len(j)] = np.array(j)
        else:
            b[i][0:len(j)] = np.zeros([1,de])
    np.save('train',b)


    #labels = [pattern.sub('', item.strip()) for sublist in labels for item in sublist if not pd.isna(item)]
    #labels = set(labels)
    #idx2labels = {string2numeric_hash(l): l for l in labels}

    return train, labels
#load_data()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sent_Dataset(Dataset):
    def __init__(self, transforms=None):
        # stuff
        ...
        self.transforms = transforms
        data = np.load('train.npy')
        data = np.transpose(data, [0,2,1])
        #data = data[:,None,:, :]
        self.data_array = data
        self.labels_array = np.load('labels.npy')

    def __getitem__(self, index):

        data =self.data_array[index]
        data = torch.FloatTensor(data, device = device)
        label = self.labels_array[index]
        label = torch.FloatTensor(label, device = device).view(1, -1)

        if self.transforms is not None:
            data = self.transforms(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return data, label

    def __len__(self):
        return len(self.data_array)

batch_size = 100
sent_dataset =  Sent_Dataset()
idx_train, idx_valid = train_test_split(range(sent_dataset.__len__()), test_size=0.1, random_state=42)
train_sampler = BatchSampler(SubsetRandomSampler(idx_train[:]), batch_size, False)
valid_sampler = BatchSampler(SubsetRandomSampler(idx_valid[:]), batch_size, False)

train_loader = DataLoader(sent_dataset, batch_sampler=train_sampler, shuffle=False, num_workers=4,
                          pin_memory=True if torch.cuda.is_available() else False)
valid_loader = DataLoader(sent_dataset, batch_sampler=valid_sampler, shuffle=False,num_workers=4,
                          pin_memory=True if torch.cuda.is_available() else False)


class _classifier(nn.Module):
    def __init__(self, nlabel):
        super(_classifier, self).__init__()
        self.conv1 = nn.Conv1d(100, 64, 3, padding=2)
        self.fc1 = nn.Linear(64*58, nlabel)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x





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
        self._accuracies.append(accuracy_score(y, y_pred))


    def compute(self):
        #if self._num_examples == 0:
            #raise NotComputableError('CategoricalAccuracy must have at least one example before it can be computed')

        return np.average(self._accuracies),np.average(self._macro_accuracies),np.average(self._micro_accuracies)


nlabel = len(sent_dataset.labels_array[0])
classifier = _classifier(nlabel)

optimizer = optim.Adam(classifier.parameters())
criterion = nn.MultiLabelSoftMarginLoss()


trainer = create_supervised_trainer(classifier, optimizer,criterion, device=device)

evaluator = create_supervised_evaluator(classifier, metrics= {'accuracy': MultilabelCategoricalAccuracy(),
                                                              'nll': Loss(criterion)}, device=device)



print(device)
log_interval = 1
epochs = 10
validate_every = 100
checkpoint_every = 100
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1
    if iter % log_interval == 0:
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.7f}"
              "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))

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


trainer.run(train_loader, max_epochs=epochs)


# for epoch in range(epochs):
#     losses = []
#     for i, inputv ,labelsv in enumerate(data_loader):
#
#         output = classifier(inputv)
#         loss = criterion(output, labelsv)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.data.mean())
#
#     print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))