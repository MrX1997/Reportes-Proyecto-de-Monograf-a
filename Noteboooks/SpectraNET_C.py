#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Jairo Andres Saavedra Alfonso
# 01 de Febrero de 2019
# Universidad de Los Andes
# Phycis 
######################__________________Weekly Report__________________######################
# Beta 1.0


# In[1]:


#Packages
from astropy.io import fits
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.utils.data
from Data_Loader import Data_Loader,Load_Files
import time
import os

cmd='jupyter nbconvert --to python SpectraNET_C.ipynb'
os.system(cmd)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Assuming that we are on a CUDA machine, this should print a CUDA device:
print('This net is brought to you by',device)


# In[3]:


N_sample=80000
batch_size=480
n_iter=10000  
test_size=0.2 # 20%
val_size=0.25 # 25% of trainning size

n_train=int(N_sample*(1-test_size)*(1-val_size))
epochs = int(n_iter / (n_train / batch_size))

fi= open("Trainning_INFO.txt","w+")

fi.write('INFO: Epochs:{} -- Batch size:{} \n'.format(epochs,batch_size))


# In[4]:



X,y=Load_Files('truth_DR12Q.fits','data_dr12.fits',N_sample, None, classification=True)
train_loader,test_loader,val_loader=Loader(X,y,N_sample)


# In[9]:


# CNN for classification
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support


learning_rate=0.1

class Net_C(nn.Module):
    def __init__(self):
        super(Net_C, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 15,stride=2)
        self.conv2 = nn.Conv1d(64, 128, 15,stride=2)
        self.conv3 = nn.Conv1d(128, 256, 15,stride=2)
        self.conv4 = nn.Conv1d(256, 256, 15,stride=2)
        self.pool = nn.MaxPool1d(2, 1)
        self.fc1 = nn.Linear(3328, 16)
        #self.fc1 = nn.Linear(10300, 16)
        self.fc2 = nn.Linear(16, 4)
        self.dropout=nn.Dropout(0.5)
        self.bn=nn.BatchNorm1d(16)


    def forward(self, x):
        in_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        ##x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))     
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x)
    


# In[10]:



net_C = Net_C()
print(net_C)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net_C.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=8, 
                                             verbose=True, threshold=0.00001, threshold_mode='rel',
                                             cooldown=1, min_lr=1e-8, eps=1e-08)

loss_=[]
loss_val=[]

iterations=0 

# Training loop
for i in range(epochs):
    fi.write('Epoch:{} \n'.format(i+1))

    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader,0):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        
        output = net_C(data)
        loss = criterion(output, target)  
        loss.backward()
        optimizer.step()
        
        running_loss = loss.item()
        
        iterations+=1
        
        if(iterations % 60 == 0):
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in val_loader:
                
                images, labels = Variable(images), Variable(labels)
                
                # Forward pass only to get logits/output
                outputs = net_C(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                loss_v = criterion(outputs, labels) 
                running_loss_val = loss_v.item()
                loss_val.append(running_loss_val)
                loss_.append(running_loss)
                

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                # Without .item(), it is a uint8 tensor which will not work when you pass this number to the scheduler
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            if(iterations % 20 == 0):
                fi.write('Batch: {} -- Iteration: {} -- Trainning Loss: {} -- Validation loss: {} -- Accuracy: {} % \n'.format(batch_idx+1, iterations, loss.item(), loss_v.item(), accuracy))
                                
    scheduler.step(accuracy)
    
print('INFO: Finished Training')

loss_=np.asarray(loss_)

epoch=np.linspace(0,len(loss_),len(loss_))

plt.plot(epoch,loss_,label='Training loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('Train_loss_Classification.jpg')
plt.close()
# In[ ]:


loss_val=np.asarray(loss_val)

epoch=np.linspace(0,len(loss_val),len(loss_val))

plt.plot(epoch,loss_val,label='Validation loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.savefig('Validation_loss_Classification.jpg')
plt.close()


# In[ ]:


correct = 0
total = 0
d=[]
d1=[]

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net_C(images)
        _, predicted = torch.max(outputs.data, 1)
        d.append(predicted)
        d1.append(labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        

print('INFO: Accuracy of the network on the test images: %d %%' % (100 * correct / total))
#d=np.asarray(d)
#print(d[0].shape)
#print(d1[0].shape)
y_pred=torch.cat((d[0],d[1]),0)
y_test=torch.cat((d1[0],d1[1]),0)
#print(y_pred.shape)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

class_names=['Star','Galaxy','QSO','QSO_BAL']

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    """
    #print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.subplots(121)
#y_test=y_test.detach().numpy()
#y_pred=y_pred.detach().numpy()
#print(y_pred)
plot_confusion_matrix(y_test, y_test, classes=class_names, title='Confusion matrix')
plt.savefig('cm_train.png')
#plt.subplots(122)
plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix')
plt.savefig('cm_test.png')

from sklearn.metrics import precision_recall_curve
p,r,f,s=precision_recall_fscore_support(y_test, y_pred, average=None)#,labels=['Star','Galaxy','QSO','QSO_BAL'])

print('Support:','Star:',round(s[0],4),'| Galaxy:',round(s[1],4),'| QSO:',round(s[2],4),'| QSO_BAL:',round(s[3],4))
print('Precision:','Star:',round(p[0],4),'| Galaxy:',round(p[1],4),'| QSO:',round(p[2],4),'| QSO_BAL:',round(p[3],4))
print('Recall:','Star:',round(r[0],4),'| Galaxy:',round(r[1],4),'| QSO:',round(r[2],4),'| QSO_BAL:',round(r[3],4))
print('F_score:','Star:',round(f[0],4),'| Galaxy:',round(f[1],4),'| QSO:',round(f[2],4),'| QSO_BAL:',round(f[3],4))

end = time.time()
print('Running time:',end - start)

rt=(end-start)/60/60
accu=100 * correct / total

fi.write('Time {} \n'.format(rt))
fi.write('Accuracy {} \n'.format(accu))
fi.write('Presicion {} \n'.format(p))
fi.write('Recall {} \n'.format(r))
fi.write('F1: {} \n'.format(f))
fi.write('Support: {} \n'.format(s))
fi.close()

