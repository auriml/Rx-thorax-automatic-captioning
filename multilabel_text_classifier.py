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
from ignite.engine import Engine, Events,  _prepare_batch
from ignite.metrics import CategoricalAccuracy, Loss
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.externals import joblib
from collections import defaultdict


from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite._utils import convert_tensor
import argparse
import remotedebugger as rd
import time
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('-pred', '--predict',  action='store_true', help='Load best model and predict')
parser.add_argument('-mc', '--model-class',  default='CNN_ATT', help='Type of model to run (RNN,CNN, CNN_ATT, RNN_ATT) ')
parser.add_argument('-mep', '--max-epoch',  type=int, default=100, help='Epoch to run')
parser.add_argument('-estop', '--early-stopping',  action='store_true', help='Early stopping')
parser.add_argument('-log', '--log',  action='store_true', help='Log metrics')
parser.add_argument('-b',   '--batch-size', type=int, default=None, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-g',   '--gpus', type=int, default=None, help='GPUs')
parser.add_argument('-l',   '--learning-rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-nbn', '--no-batchnorm', action='store_true', help='Do NOT use batch norm')
parser.add_argument('-af',  '--activation-function', default='relu', help='Activation function to use (relu|prelu), e.g. -af prelu')

parser.add_argument('-m', '--model',   help='load hdf5 model (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights from model (and continue training)')

parser.add_argument('-fc',  '--fully-connected-layers', nargs='+', type=int, default=[512], help='Specify last FC layers, e.g. -fc 1024 512 256')
parser.add_argument('-do',  '--dropout', type=float, default=0, help='Dropout rate')

parser.add_argument('-me',  '--max-emb', type=int, default=64, help='Maximum size of embedding vectors for categorical features')

parser.add_argument('-kf',   '--k-folds', type=int, default=11,    help='Evaluate model in k-folds')


parser.add_argument('-t',   '--test',                     action='store_true', help='Test on test set')
parser.add_argument('-tt',  '--test-train',               action='store_true', help='Test on train set')

rd.attachDebugger(parser)

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)

args = parser.parse_args()
args.predict = True
de = 100 # dimension word embedding

def string2numeric_hash(text):
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)


def labels2set(x):
    l = [re.sub(r' right side| left side| both side.*| bilateral', '', item) for item in x.values if pd.isnull(item) == False] #remove localizations that were manually added to labels
    pattern = re.compile('[^a-zA-Z,\s\-]+')
    s = set(pd.Series([pattern.sub('', item.strip()) for item in l if not pd.isna(item)]))
    return s
    

def load_labeled_data(de):

    train = []
    labels = []
    texts = []
    seq_lengths = []
    sent_labels = '/manual_review/labeled_sent_28K.csv'
    path = root + '/Rx-thorax-automatic-captioning' + sent_labels
    column_names = ['text','topic', 'counts']
    column_names.extend(list('123456789'))
    sent_labels = pd.read_csv(path, sep = ',',  header = 0, names =column_names)
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
#load_labeled_data(de)




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
        self.best_acc = 0
        self.best_acc_by_kfold = defaultdict(lambda: 0)
        self.train_accuracies = []
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.current_epoch = None
        self.flag_val_labels = False #Only when set to True, val_labels_by_kfold will log them 
        self.labels_by_kfold = defaultdict(lambda: None) #only those labels included both in training and validation set are learneable in a k-fold
        self.val_labels_by_kfold = defaultdict(lambda: None) #only those labels included both in training and validation set are learneable in a k-fold
        self.val_labels_precision_by_kfold = defaultdict(lambda: None) # shape(k_folds, n_labels)

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
                    batch = []
                    lengths = []
            if len(batch) > 0 and not self.drop_last:
                merge = list(zip(batch,lengths))
                merge.sort(key=lambda tup: tup[1])
                batch, _ = zip(*merge[::-1])
                yield list(batch)
                


def create_supervised_trainer(model, optimizer, loss_fn, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
        y_pred, _ = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred, y
    
    return Engine(_update)

def create_supervised_evaluator(model, metrics={}, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = _prepare_batch(batch, device=device)
            y_pred, _ = model(x)
            return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

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
        return x, None


class _classifierCNN(nn.Module):
    def __init__(self, nlabel, de, attention = False):
        super(_classifierCNN, self).__init__()
        num_filter_maps = 128
        self.conv1 = nn.Conv1d(de, 64, 3, padding=2)
        self.pooling = nn.MaxPool1d(kernel_size = 2)
        self.conv2 = nn.Conv1d(64, num_filter_maps, 3, padding=2)
        self.fc1 = nn.Linear(num_filter_maps*31, nlabel)
        self.attn = attention
        if self.attn:
            #context vectors for computing attention as in 2.2
            self.U = nn.Linear(num_filter_maps, nlabel)
            nn.init.xavier_uniform(self.U.weight)

            #final layer: create a matrix to use for the L binary classifiers as in 2.3
            self.final = nn.Linear(num_filter_maps, nlabel)
            nn.init.xavier_uniform(self.final.weight)


    def forward(self, input):
        input = input[1]
        x = F.relu(self.conv1(input))
        x = self.pooling(x)
        alpha = None
        if self.attn: 
            #apply convolution and nonlinearity (tanh)
            #x = F.tanh(self.conv2(x).transpose(1,2))
            x = F.relu(self.conv2(x).transpose(1,2))
            x = F.dropout(x, p=0.4, training=self.training)
            #apply attention
            alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
            #document representations are weighted sums using the attention. Can compute all at once as a matmul
            m = alpha.matmul(x)
            #final layer classification
            x = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        else:
            x = F.relu(self.conv2(x))
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
        return x, alpha
# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, attention = False):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*1, num_classes)  # 2 for bidirection
        self.attn = attention
        if self.attn:
            #context vectors for computing attention as in 2.2
            self.U = nn.Linear(hidden_size*2, num_classes)
            nn.init.xavier_uniform(self.U.weight)

            #final layer: create a matrix to use for the L binary classifiers as in 2.3
            self.final = nn.Linear(hidden_size*2, num_classes)
            nn.init.xavier_uniform(self.final.weight)
    
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
        alpha = None
        if self.attn: 
            #apply convolution and nonlinearity (tanh)
            #x = F.tanh(self.conv2(x).transpose(1,2))
            
            out , out_length= nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            #out = out.view(out.shape[0],out.shape[1],2,hidden_size)
            x = F.relu(out)
            x = F.dropout(x, p=0.4, training=self.training)
            #apply attention
            alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
            #document representations are weighted sums using the attention. Can compute all at once as a matmul
            m = alpha.matmul(x)
            #final layer classification
            out = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        else:
            out = ht[-1]
            out = F.dropout(out, p=0.4, training=self.training)
            out = self.fc(out)   

        return out, alpha

#Arguments: n is the precision threshold used to retrieve labels, scores is an array (n_labels) of avg scores by labels
#Output: the labels predicted with a precision above n
def precision_at_n(n, precision_scores):
    yl = list(sent_dataset.mlb.classes_)
    print(precision_scores)
    z = list(zip(yl,list(precision_scores)))
    z.sort(key=lambda tup: float('-inf') if np.isnan(tup[1]) else tup[1],reverse=True)
    #with open("labels_precision.txt", "a") as myfile:
    #    s = ''.join(str(v) for v in z)
    #    myfile.write(s)
    #    myfile.write("\nNEW EPOCH\n")
    return z[:n]
    

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
        self._weighted_accuracies = []
        self._accuracies = []
        


    def update(self, output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred).data > 0.5
        y = y.view(y.shape[0],-1)


        self._macro_accuracies.append(f1_score(y, y_pred, average='macro'))
        self._micro_accuracies.append(f1_score(y, y_pred, average='micro'))
        self._weighted_accuracies.append(f1_score(y, y_pred, average='weighted'))
        self._accuracies.append(accuracy_score(y, y_pred))
        
    def compute(self):
        #if self._num_examples == 0:
            #raise NotComputableError('CategoricalAccuracy must have at least one example before it can be computed')
        return np.average(self._accuracies),np.average(self._macro_accuracies),np.average(self._micro_accuracies),np.average(self._weighted_accuracies)

class LabelDetailedCategoricalAccuracy(Metric):
    """
    Calculates the categorical accuracy by label

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
    - `y` must be in the following shape (batch_size, ...)
    """
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        self._precisions = [] #shape (n_batchs, n_labels)
        self._y_pred = np.array([])
        self._y = np.array([])

    def update(self, output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred).data > 0.5
        y = y.view(y.shape[0],-1)
        y = y.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        if sent_dataset.flag_val_labels == True:
            j = y[0] 
            for _,v in enumerate(y):
                j = np.logical_or(v, j)
            sent_dataset.val_labels_by_kfold[k] = np.logical_or(sent_dataset.val_labels_by_kfold[k] ,j)
            print('val_labels: ')
            print (np.sum(sent_dataset.val_labels_by_kfold[k]))
            
        
        self._y_pred = y_pred if self._y_pred.size == 0 else np.concatenate([self._y_pred, y_pred], axis=0)
        self._y = y if  self._y.size == 0 else np.concatenate([self._y, y], axis=0)

    def compute(self):
        #if self._num_examples == 0:
            #raise NotComputableError('CategoricalAccuracy must have at least one example before it can be computed')
        y = np.reshape(np.array(self._y), (-1, nlabel))
        y_pred = np.reshape(np.array(self._y_pred), (-1, nlabel))
        precision_batchs = precision_score(y, y_pred, average=None)
        yl = list(sent_dataset.mlb.classes_)
        z = list(zip(yl,list(precision_batchs)))
    
        return z

if not args.predict:
    batch_size = 1024
    batch_size = 10
    n_splits = 11
    cnn_format = False
    if "CNN" in args.model_class:
        cnn_format = True
    sent_dataset =  Sent_Dataset(CNN_format=cnn_format)
    train_accuracy_scores = []
    val_accuracy_scores = []
    train_losses_scores = []
    val_losses_scores = []
    from sklearn.model_selection import KFold
    sss = KFold(n_splits=n_splits,  random_state=42)
    k = 0
    logfile = open("log_models.txt", "a")
    #for idx_train, idx_valid in sss.split(range(sent_dataset.__len__()),sent_dataset.labels_array):
    while (k==0):
        idx_train, idx_valid = train_test_split(range(sent_dataset.__len__()), test_size=0.1, random_state=42)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        if args.log == True:
            logfile.write(st + "\n")
        print("Training size:{} ".format(len(idx_train)))
        print("Validation size:{} ".format(len(idx_valid)))
        
        train_sampler = BatchSamplerOrdered(SubsetRandomSampler(idx_train[:]), batch_size, False)
        valid_sampler = BatchSamplerOrdered(SubsetRandomSampler(idx_valid[:]), batch_size, False)

        train_loader = DataLoader(sent_dataset, batch_sampler=train_sampler, shuffle=False, num_workers=4,
                            pin_memory=True if torch.cuda.is_available() else False)
        valid_loader = DataLoader(sent_dataset, batch_sampler=valid_sampler, shuffle=False,num_workers=4,
                            pin_memory=True if torch.cuda.is_available() else False)

        nlabel = len(sent_dataset.labels_array[0])
        classifier = None
        optimizer = None
        if args.model_class == "CNN":
            classifier = _classifierCNN(nlabel,de)
            optimizer = optim.Adam(classifier.parameters())
        elif args.model_class == "CNN_ATT":
            classifier = _classifierCNN(nlabel,de, attention = True)
            optimizer = optim.Adam(classifier.parameters())
        elif args.model_class == "RNN_ATT":
            num_layers = 2
            hidden_size = 128
            input_size = de
            classifier = BiRNN(input_size, hidden_size, num_layers, nlabel, attention = True).to(device)
            optimizer = optim.RMSprop(classifier.parameters(), lr=0.01)
        elif args.model_class == 'RNN':
            num_layers = 2
            hidden_size = 128
            input_size = de
            classifier = BiRNN(input_size, hidden_size, num_layers, nlabel).to(device)
            optimizer = optim.RMSprop(classifier.parameters(), lr=0.01)
        
        criterion = nn.MultiLabelSoftMarginLoss()


        trainer = create_supervised_trainer(classifier, optimizer,criterion, device=device)

        evaluator = create_supervised_evaluator(classifier, metrics= {'accuracy': MultilabelCategoricalAccuracy(),
                                                                'label_accuracy': LabelDetailedCategoricalAccuracy(),
                                                                'nll': Loss(criterion)}, device=device)
        evaluator_training = create_supervised_evaluator(classifier, metrics= {'accuracy': MultilabelCategoricalAccuracy(),
                                                                'label_accuracy': LabelDetailedCategoricalAccuracy(),
                                                                'nll': Loss(criterion)}, device=device)



        print(device)
        log_interval = 10
        log_view_interval = 10
        epochs = args.max_epoch
        early_stopping = args.early_stopping
        validate_every = 100
        checkpoint_every = 100

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1
            if iter % log_interval == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.7f}"
                    "".format(engine.state.epoch, iter, len(train_loader), engine.state.output[0]))


        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_view(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1
            _y = engine.state.output[2]
            _y_pred = engine.state.output[1]
            if engine.state.epoch  == 1:
                y = _y.view(_y.shape[0],-1)
                y = y.cpu().detach().numpy()
                j = y[0] 
                for _,v in enumerate(y):
                    j = np.logical_or(v, j)
                sent_dataset.labels_by_kfold[k] = np.logical_or(sent_dataset.labels_by_kfold[k] , j)
            if iter % log_view_interval == 0:
                #for idx in engine.state.batch[0][0]:
                #    print(sent_dataset.texts_array[idx])
                #    print(sent_dataset.mlb.inverse_transform(np.array([sent_dataset.labels_array[idx]])))
                y_pred = torch.sigmoid(_y_pred).data > 0.5
                y = _y.view(_y.shape[0],-1)
                yl_pred = sent_dataset.mlb.inverse_transform(y_pred.cpu().detach().numpy())
                yl = sent_dataset.mlb.inverse_transform(y.cpu().detach().numpy())
                z = list(zip(yl,yl_pred))
                #print("ground_truth,predicted")
                #print(z)

                    #print("Epoch[{}] Iteration[{}/{}] Loss: {:.7f}"
                    #  "".format(engine.state.epoch, iter, len(train_loader.batch_sampler), engine.state.output))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            evaluator_training.run(train_loader)
            metrics = evaluator_training.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            s = "{}_{} {} Train Results:  Avg accuracy: {} Avg loss: {:.7f}".format(k, engine.state.epoch, args.model_class, avg_accuracy, avg_nll)
            if args.log == True:
                logfile.write(s + "\n")
            print(s)
            sent_dataset.train_accuracies.append(avg_accuracy)
            sent_dataset.train_losses.append(avg_nll)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            if engine.state.epoch == 1: 
                #labels are imbalanced and highly sparse as a result not all labels are seen on training and validation for each k-fold (only aprox 145 out of 213 are seen on validation on each k-fold)
                # the existence of single instances for some label precludes using stratified shuffling for splits
                # To allow estimating the precision for the maximum set of labels we need to use estimations from all k-folds 
                # We follow the following approach:
                sent_dataset.flag_val_labels = True
                evaluator.run(valid_loader)
                # for each k-fold, use the intersection of the labels sets for the training and validation sets (taking the set of any epoch (e.g first one) which is equal in the same k-fold)
                sent_dataset.labels_by_kfold[k] = np.logical_and(sent_dataset.labels_by_kfold[k], sent_dataset.val_labels_by_kfold[k])
                sent_dataset.flag_val_labels = False 
            else:
                evaluator.run(valid_loader)
            
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            label_accuracy = metrics['label_accuracy']
            avg_nll = metrics['nll']
            s = "{}_{} {} Valid Results:  Avg accuracy: {} Avg loss: {:.7f}".format(k, engine.state.epoch, args.model_class, avg_accuracy, avg_nll)
            if args.log == True:
                logfile.write(s + "\n")
            print(s)
            sent_dataset.val_accuracies.append(avg_accuracy)
            sent_dataset.val_losses.append(avg_nll)
            # remember best avg accuracy 
            if avg_accuracy[0] > sent_dataset.best_acc_by_kfold[k]:
                sent_dataset.best_acc_by_kfold[k] = max(avg_accuracy[0], sent_dataset.best_acc_by_kfold[k])
                #out_path = "model_epoch_{}.pth".format(engine.state.epoch) #TODO: pending add model name
                #torch.save(classifier.state_dict, out_path) 
                # log for each k-fold the precision scores for each label in the validation set for EPOCH with highest accuracy
                label_accu = sent_dataset.labels_by_kfold[k] * np.array(label_accuracy)[:,1]
                label_accu = np.vectorize(lambda x: float(x) if x!='' else np.nan)(label_accu)
                sent_dataset.val_labels_precision_by_kfold[k] = label_accu
                
                
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_view(engine):
            #evaluator.run(valid_loader)
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
        
        def score_function_f1(engine):
            weighted_f1 = engine.state.metrics['accuracy'][3]
            return weighted_f1
        def score_function_micro_f1(engine):
            micro_f1 = engine.state.metrics['accuracy'][2]
            return micro_f1
        def score_function(engine):
            loss = engine.state.metrics['nll']
            return -loss
        if early_stopping:
            handler = EarlyStopping(patience=50, score_function=score_function, trainer=trainer)
            evaluator.add_event_handler(Events.COMPLETED, handler)
        
            handler = ModelCheckpoint('./labeling_models', str(k), score_function=score_function_micro_f1,  score_name='micro_f1', n_saved=1, create_dir=True, require_empty=False)
            evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, {args.model_class: classifier})
        
        trainer.run(train_loader, max_epochs=epochs)
        k += 1

    if early_stopping: #log precision matrix by label only on the event of max accuracy using early_stopping
        with open("labels_precision_matrix.txt", "a") as myfile:
            s = list(sent_dataset.val_labels_precision_by_kfold.values())

            #s = " ".join(s)
            #myfile.write(s)
            m = np.array(s)
            precision_labels = np.nanmean(m,axis=0)
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            myfile.write(st + "\n")
            s = args.model_class + " " + "Avg Precision by Labels with number of k-folds = {}".format(n_splits)
            myfile.write(s)
            myfile.write('\n'.join([str(i) for i in precision_at_n(nlabel,precision_labels)]))
            myfile.write(str(m.shape))
            myfile.write(str(m))


        

    #plot learning curves using a fixed num of epochs for all k-folds (early stopping should be False)
    if not early_stopping:
        train_accuracy_scores = np.reshape(np.array(list(sent_dataset.train_accuracies)), (epochs,n_splits,-1), order='F')
        val_accuracy_scores= np.reshape(np.array(list(sent_dataset.val_accuracies)), (epochs,n_splits,-1), order='F')
        train_losses_scores= np.reshape(np.array(list(sent_dataset.train_losses)), (epochs,n_splits,-1), order='F')
        val_losses_scores= np.reshape(np.array(list(sent_dataset.val_losses)), (epochs,n_splits,-1), order='F')
        #print(train_accuracy_scores)
        #print(val_accuracy_scores)

        train_scores_mean = []
        train_scores_std = []
        test_scores_mean = []
        test_scores_std = []
        for e in range(epochs):
            train_scores_mean.append(train_accuracy_scores[e,:][:,2].mean())
            train_scores_std.append(train_accuracy_scores[e,:][:,2].std())
            test_scores_mean.append(val_accuracy_scores[e,:][:,2].mean())
            test_scores_std.append(val_accuracy_scores[e,:][:,2].std())

        #print(train_scores_mean)
        #print(train_scores_std)

        #print(test_scores_mean)
        #print(test_scores_std)
        import matplotlib
        matplotlib.use('Agg')
        import plot as pl

        title = "Accuracy_Curves"
        title = args.model_class + "_" + title
        pl.plot_val_curve(title, epochs = epochs, logX=False,
                            test_scores_mean = test_scores_mean,test_scores_std = test_scores_std,
                            train_scores_mean = train_scores_mean,train_scores_std = train_scores_std)


#predict
else:
    
    textFile = "sentences_preprocessed.csv" 
    path = root + '/Rx-thorax-automatic-captioning' + textFile
    df = pd.read_csv(textFile , keep_default_na=False, header = 0)
    df0 = df
    unique_reports = df['codigoinforme'].unique()
    print(len(unique_reports))

    #Convert the set of unique sentences to vectors 
    unique_sentences = df['text'].unique()
    unique_sentences_size = len(unique_sentences)
    print(unique_sentences_size)
    unique_sentences_max_length = len(max(unique_sentences,key = lambda x: len(x)))

    #load pretrained wordembedding model
    from fastText import load_model #TODO: retrain wordvectors with masa not mas (file: report_sentences_preprocessed)
    path = root + '/Rx-thorax-automatic-captioning' + '/embeddings/fasttext/text.bin'
    f = load_model(path)
    
    #load labels decoder
    mlb = joblib.load('mlb.pkl')

    #load model
    model_name = "0_RNN_ATT_41_micro_f1=0.9241743.pth"
    path = './labeling_models/' + model_name
    model = BiRNN
    model= torch.load(path, map_location={'cuda:0': 'cpu'})
    if device: model.to(device)

    
    
    
    
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    batch = 1000
    batch_number = 0
    for batch_sentences in chunks(unique_sentences,batch):
        print("batch: " + str(batch_number))
        batch_number += 1
        sent_vectors=[]
        seq_lengths = []
        texts = []
        
        b = np.zeros([len(batch_sentences),len(max(batch_sentences,key = lambda x: len(str(x).split(' ')))), de])
        batch_input = None
        for i,j in enumerate(batch_sentences):
            if len(j) > 0:
                words = str(j).split(' ')
                b[i][0:len(words)] = np.array([f.get_word_vector(w) for w in words ])
                seq_lengths.append(len(words))
            else:
                b[i][0:len(words)] = np.zeros([1,de])
                seq_lengths.append(0)
            texts.append(j)    
        
        
        #order batch by sentence length
        zi = list(zip(texts,b,seq_lengths))
        sortedzi = sorted(zi, key=lambda s: s[2],reverse=True)
        unzi = list(zip(*sortedzi))
        texts = unzi[0]
        b = torch.FloatTensor(unzi[1], device = device)
        seq_lengths = torch.tensor(unzi[2], device = device)
        #seq_length in first argument is redundant (not used in prediction)
        batch_input = (seq_lengths,b,seq_lengths)

        #convert to tensor
        batch_input = convert_tensor(batch_input, device = device)


        model.eval()
        with torch.no_grad():
            #x, y = _prepare_batch(batch, device=device)
            y_pred, _ = model(batch_input)
            y_pred = torch.sigmoid(y_pred).data > 0.5
            yl_pred = mlb.inverse_transform(y_pred.cpu().detach().numpy())
            
            z = list(zip(texts,yl_pred))
            sent_labels = pd.DataFrame(list(z), columns = ['text', 'labels'])
            df = pd.merge( df, sent_labels, how='left', on= 'text')
            
            #print(z)

            
            z_coded= [[_x,*_y,] for _x,_y in zip(texts,y_pred.cpu().numpy())]
            sent_labels_coded = pd.DataFrame(z_coded)
            sent_labels_coded = sent_labels_coded.rename(columns={ sent_labels_coded.columns[0]: "text" })
            df0 = pd.merge( df0, sent_labels_coded, how='left', on= 'text')
            #print(z_coded)
            
    
    df.to_csv('_sentences_reports_aut_labeled.csv')
    df0.to_csv('_sentences_reports_aut_labeled_coded.csv')

def generateRandomTestSample(n = 1000):
    #load sentence file
    path = root + '/Rx-thorax-automatic-captioning/' + 'sentences_reports_aut_labeled.csv'
    df = pd.read_csv(path, sep = ',' ,dtype = str)
    df = df[(df.codigoinforme.astype(int) > 3200000) & (df.codigoinforme.astype(int) < 4750000)]
    set_all = set(df.text.values)
    #exclude sentences used in training set
    sent_labels = '/manual_review/labeled_sent_28K.csv'
    path = root + '/Rx-thorax-automatic-captioning' + sent_labels
    column_names = ['text','topic', 'counts']
    column_names.extend(list('123456789'))
    training = pd.read_csv(path, sep = ',',  header = 0, names =column_names)
    set_training = set(training.text.values)
    set_test = set_all - set_training

    #choose n random senteces
    import random
    random.seed(1234)
    test_1K = random.sample(set_test, n)
    test_1K = pd.DataFrame({'text':test_1K})

    #load pred labels
    sentence_test_1K = pd.merge(df,test_1K, on='text', how= 'right')
    sentence_test_1K.drop_duplicates(subset='text', inplace=True)
    print(sentence_test_1K.shape)

    sentence_test_1K['labels'] = sentence_test_1K[sentence_test_1K.columns[3:]].apply(lambda x: "".join(x.dropna().values),axis=1)
    sentence_test_1K['labels'] = sentence_test_1K.labels.str.replace(r'(,\)|[()\'\"])','').str.split(',')


    #save sentence_test_1K.csv 
    sentence_test_1K.to_csv('sentences_test_1K.csv', columns= ['text','labels'])
    return

def testRandomSample(n = 501):
    mlb = joblib.load('mlb.pkl') #load coder/decoder from labels to one-hot and viceversa
    
    #load labeled sample test
    file = '/manual_review/labeled_test_500.csv' 
    path = root + '/Rx-thorax-automatic-captioning' + file
    sent_labels = pd.read_csv(path, sep = ',',  header = 0,  nrows = int(n))
    sent_labels['labels_set'] = sent_labels[sent_labels.columns[1:]].apply(lambda x: labels2set(x), axis=1)
    print(sent_labels['labels_set'].head())


    #load predicted labels sample test
    file = '/_sentences_test_1K.csv'
    path = root + '/Rx-thorax-automatic-captioning' + file
    sent_labels_pred = pd.read_csv(path, sep = ',',  header = 0,  nrows = int(n), )
    sent_labels_pred['labels_set_pred'] = sent_labels_pred[sent_labels_pred.columns[1:]].apply(lambda x: labels2set(x), axis=1)
    print(sent_labels_pred['labels_set_pred'].head())
    
    merge = pd.merge(sent_labels, sent_labels_pred, how='left', on= 'text')
    merge.drop_duplicates(inplace=True, subset = 'text')
    print(merge['labels_set_pred'].head())

    
    #convert to vector
    y= merge['labels_set'].apply(lambda x: list(mlb.transform([x])[0]))
    y_pred= merge['labels_set_pred'].apply(lambda x: list(mlb.transform([x])[0]))
    
    y_pred = np.vstack( y_pred)
    y = np.vstack( y )
    
    print(y_pred.shape)
    print(y.shape)

    #compute metrics
    metrics = (f1_score(y, y_pred, average='macro'),
    f1_score(y, y_pred, average='micro'),
    f1_score(y, y_pred, average='weighted'),
    accuracy_score(y, y_pred))
    print (metrics)

    return metrics 