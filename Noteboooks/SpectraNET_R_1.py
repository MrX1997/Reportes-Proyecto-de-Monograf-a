#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Jairo Andres Saavedra Alfonso
# 01 de Febrero de 2019
# Universidad de Los Andes
# Phycis 
######################__________________Weekly Report__________________######################
# Beta 1.0


# In[45]:


#Packages
from astropy.io import fits
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from torch.autograd import Variable
import torch.utils.data
import time
import os
from Data_Loader import Data_Loader,Load_Files
from matplotlib.gridspec import GridSpec


#cmd='jupyter nbconvert --to python SpectraNET_R.ipynb'
#os.system(cmd)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print('This net is brought to you by',device)


N_sample=80000
batch_size=480
n_iter=10000

test_size=0.2 # 20%
val_size=0.25 # 25% of trainning size

n_train=int(N_sample*(1-test_size)*(1-val_size))
epochs = int(n_iter / (n_train / batch_size))

f= open("Trainning_INFO_Regression_80k_QSO.txt","w+")

f.write('INFO: Epochs:{} -- Batch size:{} \n'.format(epochs,batch_size))

start=time.time()

X,y=Load_Files('truth_DR12Q.fits','data_dr12.fits',N_sample,['QSO'],classification=False)
train_loader,test_loader,val_loader=Data_Loader(X, y, N_sample, batch_size,test_size, val_size)



"""
for i in range(100):
    x=np.linspace(300,1000,443)
    print('Redshift:{}'.format(y[i]))
    plt.plot(x,X[i,:])
    plt.xlabel('Wavelength')
    plt.ylabel('Renormalized Flux [Arb unix]')
    plt.show()
    
"""


# In[13]:


class Net_R(nn.Module):
    def __init__(self):
        super(Net_R, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 10,stride=2)
        self.conv2 = nn.Conv1d(64, 128, 10,stride=2)
        self.conv3 = nn.Conv1d(128, 256, 10,stride=2)
        self.conv4 = nn.Conv1d(256, 256, 10,stride=2)
        self.pool = nn.MaxPool1d(2, 1)
        self.fc1 = nn.Linear(4608, 128)
        self.bn=nn.BatchNorm1d(128)
        #self.fc1 = nn.Linear(10300, 16)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout=nn.Dropout(0.5)


    def forward(self, x):
        in_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))     
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    


# In[14]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support

learning_rate=0.01
log_interval=10


net_R = Net_R()
print('INFO: SpectraNET for regression: {}'.format(net_R))

optimizer = torch.optim.SGD(net_R.parameters(), lr=learning_rate) #for Rgrss
loss_func = torch.nn.MSELoss()


loss_=[]
def train(epoch):
    #model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader,0):
        data, target = Variable(data), Variable(target)
       
        optimizer.zero_grad()
        
        output = net_R(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss = loss.item()
        loss_.append(running_loss)
        f.write('Batch: {} <--> Loss: {} \n'.format(batch_idx+1,running_loss))
        #if(batch_idx !=0):    # print every 2000 mini-batches
            #print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 1000))
            #running_loss = 0.0
    


# Training loop
for i in range(epochs):
    f.write('Epoc: {}'.format(i+1))
    train(i)
    
    
print('Finished Training')

loss_=np.asarray(loss_)

epoch=np.linspace(0,len(loss_),len(loss_))

plt.plot(epoch,loss_,label='Training loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Train Loss - Regression')
plt.legend()
plt.savefig('Training_Loss_Regression_80k_QSO.jpg')
plt.close()


# In[16]:


d=[]
d1=[]

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net_R(images)
        #_, predicted = torch.max(outputs.data, 1)
        #print(predicted,predicted.shape)
        outputs=outputs.view(outputs.size(0),-1)
        d.append(outputs)
        d1.append(labels)

d=d[0]
d=d.reshape(1,-1)

y_pred=d[0]
y_test=d1[0]


y_pred=torch.Tensor.numpy(y_pred)
y_test=torch.Tensor.numpy(y_test)

x = y_test
y = y_pred

fig = plt.figure(figsize=(5,5 ))

gs = GridSpec(4,4)

ax_joint = fig.add_subplot(gs[1:4,0:3])
ax_marg_x = fig.add_subplot(gs[0,0:3])
ax_marg_y = fig.add_subplot(gs[1:4,3])
ax_joint.scatter(x,y)
ax_marg_x.hist(x,70,density=True)
ax_marg_y.hist(y,70,density=True,orientation="horizontal")

# Turn off tick labels on marginals
plt.setp(ax_marg_x.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)

# Set labels on joint
ax_joint.set_xlabel('Z_in')
ax_joint.set_ylabel('Z_out')
plt.legend()
plt.savefig('Z_in_vs_Z_out_80k_QSO.jpg')
plt.close()

AE=(abs(y_pred-y_test)*100)/(y_test)

plt.scatter(y_test,AE,label='Redshift')
plt.xlabel('Z_in')
plt.ylabel('Absolute Error')
plt.title('Absolute Error')
plt.legend()
plt.savefig('AE_vs_Z_in_80k_QSO.jpg')
plt.close()

MSE_Test=mean_squared_error(y_pred,y_test)
MAE_Test=mean_absolute_error(y_pred,y_test)
R2_Test=r2_score(y_pred,y_test)

f.write('MSE: {} \n'.format(MSE_Test))
f.write('MAE: {} \n'.format(MAE_Test))
f.write('R2: {} \n'.format(R2_Test))


end=time.time()
df=end-start
f.write('Time: {} \n'.format(df))

f.write('y pred: {}\n'.format(y_pred))
f.write('y test: {}\n'.format(y_test))

f.close()



#cmd='shutdown 0'
#os.system(cmd)
