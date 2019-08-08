#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Jairo Andres Saavedra Alfonso
# 01 de Febrero de 2019
# Universidad de Los Andes
# Phycis 
######################__________________Report 01__________________######################


# In[1]:


#Packages
from astropy.io import fits
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from astropy.table import Table
get_ipython().system('jupyter nbconvert --to python Data_Loader_01.ipynb')


# In[3]:

def Load_Files(file_1,file_2);
	hdul = fits.open(file_1) # Open file 'truth_DR12Q.fits'
	info=hdul.info() # File info
	columns=hdul[1].columns # File Columns 
	print(info,'/n',columns)
	data=hdul[1].data # Database of spectra with human-expert classifications 


	# Reading data from data_dr12.fits. This file had the spectra from data dr12. 
	hdul_2 = fits.open(file_2) # Open file
	info=hdul_2.info() # File info 
	columns=hdul_2[1].columns # File Columns		 
	print(hdul,'/n',columns)
	data2=hdul_2[1].data # Database of spectra
	spectra=hdul_2[0].data # Spectrum of each object 


# In[4]:


######################__________________Report 02__________________######################

## This week I pretend to find some correaltions betwen objects with human-expert classification and spretum from DR12

# Subset of PLATE parameters of both data
data_PLATE_1=data['PLATE']
data_PLATE_2=data2['PLATE']

# Subset of MJD parameters of both data
data_MJD_1=data['MJD']
data_MJD_2=data2['MJD']

# Subset of FIBERID parameters of both data
data_FIBERID_1=data['FIBERID']
data_FIBERID_2=data2['FIBERID']

# Subset of FIBERID parameters of both data
data_ID_1=data['THING_ID']
data_ID_2=data2['TARGETID']

# I make here an intersecting set for all three parameters (PLATE, MJD, FIBERID) in both data.
data_PLATE_CO=np.intersect1d(data_PLATE_1,data_PLATE_2)
data_MJD_CO=np.intersect1d(data_MJD_1,data_MJD_2)
data_FIBERID_CO=np.intersect1d(data_FIBERID_1,data_FIBERID_2)
data_ID_CO=np.intersect1d(data_ID_1,data_ID_2)

# As we can see, in both database, there is a correlation betwen the number of Plates, the modified julian day and the Fiber ID. 
print('Number of Plates use in both datasets:',data_PLATE_CO.shape)
print('Number of MJD use in both datasets:',data_MJD_CO.shape)
print('Number of FIBERID use in both datasets:',data_FIBERID_CO.shape)
print('Number of FIBERID use in both datasets:',data_ID_CO.shape)
#print(data_PLATE_1.dtype)


# In[5]:


# The Spectra in this database have three main parameters: Plate ID, The modified julian day and fiber ID of the observation.
# Let's take a look of the first spectrum.
x=np.linspace(360,1000,443) # I cut the sample to 443 pixels in spaced log-wavelength.  
#x=np.linspace(0,886,886)
zero_spectrum=spectra[12030] #First spectrum
zero_spectrum=zero_spectrum[:443]

PLATE=data2['PLATE'] # Spectra's Plate ID
MJD=data2['MJD'] # Spectra's the modified juliam day
FIBERID=data2['FIBERID'] # Spectra's fiber ID

zero_plate=PLATE[0] # zero spectrum Plate ID
zero_mjd=MJD[0] # zero spectrum MJD
zero_fiberid=FIBERID[0] # zero spectrum Fiber ID
param = 'Plate = {:.2f}, mjd = {:.2f}, fiberid={:.2f}'.format(zero_plate, zero_mjd, zero_fiberid)
plt.plot(x,np.log(zero_spectrum))
plt.xlabel('Wavelength [nm]')
plt.ylabel('Log-Flux [arb. units]')
plt.title(param)
plt.xlim([360,1000])
plt.savefig('spectrum.png')
plt.grid()

# I noticed that each object and spectrum don't have the same label. So it's imposible to make some ID correlations.  
print(data2['TARGETID'])
print(data['THING_ID'])
da=np.intersect1d(data2['TARGETID'],data['THING_ID'])
print(da.shape)

# So, in orden to make a correlation betwen identified object and spectrum we need to use all three parameters (Plate ID, MJD, FiberID)


# In[6]:


# The column 'CLASS_PERSON' have a class identifier for each spectrum: STARS=1, GALAXY=4, QSO=3 and QSO_BAL=30.
C_P=data['CLASS_PERSON'] #Class Person column 
STAR=C_P[C_P==1] # objects classified as stars
GALAXY=C_P[C_P==4] # objects classified as galaxies 
QSO=C_P[C_P==3] # objects classified as QSO (Quasars)
QSO_BAL=C_P[C_P==30] # objects classified as QSO BAL (Quasars with Broad Absortions Lines)
N_C=C_P[C_P!=30]   
N_C=N_C[N_C!=3]
N_C=N_C[N_C!=1]
N_C=N_C[N_C!=4] # objects wrong classified
print('Star:',STAR.shape)
print('Galaxy:',GALAXY.shape)
print('QSO:',QSO.shape)
print('QSO BAL:',QSO_BAL.shape)
print('NN:',N_C.shape)


# In[7]:


Z_VI=data['Z_VI'] # Redshift of each object
print(Z_VI[Z_VI==0.0].shape)
Z_C_P=data['Z_CONF_PERSON']
print(Z_C_P[Z_C_P==0].shape)
T_ID=data['THING_ID']
i=T_ID==-1
T_ID=T_ID[i]


# In[8]:


ii=C_P==3
oo=C_P==30
Z_VI_QSO=Z_VI[ii]
Z_VI_QSO_BAL=Z_VI[oo]
plt.hist(Z_VI_QSO,100,density=True)
plt.xlabel('Redshift')
plt.title('QSO')


# In[9]:


plt.hist(Z_VI_QSO_BAL,100,density=True)
plt.xlabel('Redshift')
plt.title('QSO_BAL')


# In[10]:


print(data2.shape)


# In[11]:


# I create two DataFrame for Superset_DR12Q and data_dr12 with only three parameters
data={'PLATE':data_PLATE_1,'MJD':data_MJD_1,'FIBERID':data_FIBERID_1,'ID':data_ID_1}
data=pd.DataFrame(data=data)

data2={'PLATE':data_PLATE_2,'MJD':data_MJD_2,'FIBERID':data_FIBERID_2,'ID':data_ID_2}
data2=pd.DataFrame(data=data2)


# In[12]:


# I convert all objects in both set to string chain in orden to combine them as one new ID.
data['PLATE']=data['PLATE'].astype(str)
data['MJD']=data['MJD'].astype(str)
data['FIBERID']=data['FIBERID'].astype(str)
#data['ID']=data['ID'].astype(str)


data['PM'] = data['MJD'].str.cat(data['FIBERID'],sep="-")
#data['M'] = data['FIBERID'].str.cat(data['ID'],sep="-")

data['NEWID'] = data['PLATE'].str.cat(data['PM'],sep="-")
data_1=data.drop(columns=['PLATE','MJD','FIBERID','ID','PM']).values # New set of database 2 with new ID's
print(data_1.dtype)

data2['PLATE']=data2['PLATE'].astype(str)
data2['MJD']=data2['MJD'].astype(str)
data2['FIBERID']=data2['FIBERID'].astype(str)
#data2['ID']=data2['ID'].astype(str)


data2['PM'] = data2['MJD'].str.cat(data2['FIBERID'],sep="-")
#data2['M'] = data2['FIBERID'].str.cat(data2['ID'],sep="-")

data2['NEWID'] = data2['PLATE'].str.cat(data2['PM'],sep="-")
data_2=data2.drop(columns=['PLATE','MJD','FIBERID','ID','PM']).values # New set of database 2 with new ID's
print(data_2.shape)
p=data_2=='4869-55896-132'
ll=data_2[p]
print(ll)


# In[13]:


# With the routine of numpy intersect1d, I find the intersections elements in both sets. This elements  
data_CO=np.array(np.intersect1d(data_1,data_2,return_indices=True))

data_CO_objects=data_CO[0] # The unique new ID of each element in both sets
data_CO_ind1=data_CO[1] # Indices of intersected elements from the original data 1 (Superset_DR12Q.fits) 
data_CO_ind2=data_CO[2] # Indices of intersected elements form the original data 2 (data_dr12.fits)
print('I find',len(data_CO_objects),'objects with spectra from DR12')
print(data_CO_ind1,data_CO_ind2)
indi={'ind1':data_CO_ind1,'ind2':data_CO_ind2}
ind=pd.DataFrame(data=indi,index=data_CO_ind1)


"""
mflux = np.ma.average(spectra[:,:443], weights=spectra[:,443:],axis=1)
sflux = np.ma.average((spectra[:,:443]-mflux[:,None])**2, weights=flux[:,443:], axis=1)
sflux = np.sqrt(sflux)
spectra = (flux[:,:443]-mflux[:,None])/sflux[:,None]
spec= pd.DataFrame(spectra)

print(spec.shape)
print(ind)
"""


# In[ ]:


# Now that I know which object have a spectrum. I can make a unique database of objects
#hdul = fits.open('truth_DR12Q.fits')
#hdul2 = fits.open('data_dr12.fits')
#data=hdul[1].data
#info=hdul[1].columns

#ti=np.array(data['THING_ID'],dtype=float)
#pl=np.array(data['PLATE'],dtype=float)
#mjd=np.array(data['MJD'],dtype=float)
#fid=np.array(data['FIBERID'],dtype=float)
cp=np.array(data['CLASS_PERSON'],dtype=float)
z=np.array(data['Z_VI'],dtype=float)
zc=np.array(data['Z_CONF_PERSON'],dtype=float)
bal=np.array(data['BAL_FLAG_VI'],dtype=float)
bi=np.array(data['BI_CIV'],dtype=float)

d={'CLASS_PERSON':cp,'Z_VI':z,'Z_CONF_PERSON':zc,'BAL_FLAG_VI':bal,'BI_CIV':bi}
data_0=pd.DataFrame(data=d)#.values #super database
obj=data_0.loc[data_CO_ind1]

print(obj.shape)


# In[ ]:


######################__________________Report 03__________________######################


# Balance of classes 
C_P=obj['CLASS_PERSON'] #Class Person column 
STAR=C_P[C_P==1] # objects classified as stars
GALAXY=C_P[C_P==4] # objects classified as galaxies 
QSO=C_P[C_P==3] # objects classified as QSO (Quasars)
QSO_BAL=C_P[C_P==30] # objects classified as QSO BAL (Quasars with Broad Absortions Lines)
N_C=C_P[C_P!=30]   
N_C=N_C[N_C!=3]
N_C=N_C[N_C!=1]
N_C=N_C[N_C!=4] # objects wrong classified
print('Stars:',STAR.shape)
print('Galaxies:',GALAXY.shape)
print('QSO:',QSO.shape)
print('QSO BAL:',QSO_BAL.shape)
print('No class:',N_C.shape)


# In[ ]:


# Preprocessing. I remove non-classified objects also objects with negative redshift.  
stars=obj.loc[obj['CLASS_PERSON']==1]
galaxies=obj.loc[obj['CLASS_PERSON']==4]
qsos=obj.loc[obj['CLASS_PERSON']==3]
qsos_bal=obj.loc[obj['CLASS_PERSON']==30]

frames=[stars,galaxies,qsos,qsos_bal]
new_obj=pd.concat(frames)#, keys=['stars', 'galaxies', 'qso','qso_bal'])

#new_obj=new_obj.loc[new_obj['Z_VI']!=0]
obj=new_obj.loc[new_obj['Z_CONF_PERSON']!=0]
#indio=np.array(obj.index)
#for i in range(len(indio)):
#    print(indio[i])
print(obj.shape)


# In[ ]:


# Sample of objects. I chosen 2500 object per class. 
stars=obj.loc[obj['CLASS_PERSON']==1]
galaxies=obj.loc[obj['CLASS_PERSON']==4]
qsos=obj.loc[obj['CLASS_PERSON']==3]
qsos_bal=obj.loc[obj['CLASS_PERSON']==30]

N_sample=20000
sample_star=stars.sample(n=int(N_sample/4),weights='CLASS_PERSON', random_state=5)
sample_galaxy=galaxies.sample(n=int(N_sample/4),weights='CLASS_PERSON', random_state=5)
sample_qso=qsos.sample(n=int(N_sample/4),weights='CLASS_PERSON', random_state=5)
sample_qso_bal=qsos_bal.sample(n=int(N_sample/4),weights='CLASS_PERSON', random_state=5)

sample_objects=pd.concat([sample_star,sample_galaxy,sample_qso,sample_qso_bal])

ind_star=np.array(sample_star.index)
ind_galaxy=np.array(sample_galaxy.index)
ind_qso=np.array(sample_qso.index)
ind_qso_bal=np.array(sample_qso_bal.index)

indi=np.concatenate((ind_star, ind_galaxy,ind_qso,ind_qso_bal), axis=None)
indi1=ind.loc[indi].values
#print(indi)

spectra_=np.zeros((N_sample,443))
j=0
for i in indi:
    k=indi1[j,1]
    spectra_[j,:]=np.log(abs(spectra[k,:443]))
    j=j+1    
spectra_=pd.DataFrame(spectra_)
X=spectra_.replace(-np.inf,0)


X=X.values

y=sample_objects['CLASS_PERSON']
y=y.replace([1, 4, 3, 30], [0,1,2,3]).values
y=np.array(y,dtype=float)
print(X.shape,y.shape) 


# In[26]:


######################__________________Report 05__________________######################

# my first Neural Network. SpectraNET :}

import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
"""
X_train = Variable(torch.tensor([[X_train]], dtype=torch.float))
X_test = Variable(torch.tensor([X_test], dtype=torch.float))
y_train = Variable((torch.tensor(y_train, dtype=torch.long)))
y_test = Variable((torch.tensor(y_test, dtype=torch.long)))
print(X_train.shape,y_train.shape)
#print(X_train)
# Scaling 
X_train_max, _ = torch.max(X_train, 0)
X_test_max, _ = torch.max(X_test, 0)
#print(X_train_max)
X_train = torch.div(X_train, X_train_max)
X_test = torch.div(X_test, X_test_max)

#y_train= torch.div(y_train, 100)
#y_test= torch.div(y_train, 100)
#y_test1=torch.tensor(X_test[0,:].reshape(1,-1),dtype=torch.float)
"""


# In[27]:


import torch.utils.data
batch_size=2000
"""
tl = torch.utils.data.TensorDataset(X_train, y_train)
train_loader=torch.utils.data.DataLoader(tl, batch_size=200, shuffle = True)
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape,labels.shape)
"""
train_data = []
for i in range(y_train.shape[0]):
    xt=X_train[i,:].reshape(1,-1)
    train_data.append([Variable(torch.tensor([xt], dtype=torch.float)), y_train[i]])
    
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
i1, l1 = next(iter(train_loader))
print(i1.shape,l1.shape)


# In[28]:


test_data = []
for i in range(y_test.shape[0]):
    xtst=X_test[i,:].reshape(1,-1)
    test_data.append([Variable(torch.tensor([xtst], dtype=torch.float)), y_test[i]])
    
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
i1, l1 = next(iter(test_loader))
print(i1.shape,l1.shape)


# In[ ]:





# In[29]:


# Implementation
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support

learning_rate=0.01

epoc=10
log_interval=10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2)
        self.conv2 = nn.Conv2d(16, 16, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(110, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.shape)
        return F.log_softmax(x)
    
net = Net()
print(net)
"""
# create a stochastic gradient descent optimizer
opt = optim.Adam(params=net.parameters(), lr=learning_rate)
# create a loss function
criterion = nn.CrossEntropyLoss()
LOSS=[]
p=[]
r=[]
f=[]

p1=[]
r1=[]
f1=[]

p2=[]
r2=[]
f2=[]

p3=[]
r3=[]
f3=[]

dataiter = iter(trainloader)
#print(dataiter.shape)
images, labels = dataiter.next()
print(images.shape)
images=images[0,0]
print(images.shape)
# show images
plt.imshow(images)
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        print(inputs.shape)
        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
def train(epoch):
    #model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        print(data.shape,target.shape)
        #images=data[batch_idx,0]
        #plt.imshow(images)
        #plt.show()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')
        
for i in range(epoc):
    train(i)


# In[ ]:


"""
for i in range(epochs):
    

    opt.zero_grad()
    data=X_train
    net_out = net(data)
    print(net_out)
    #print('loss', loss.detach().item())       
    #print(net_out.shape,y_train)
    loss = criterion(net_out, y_train)
    LOSS.append(loss.detach().item())
    loss.backward()
    opt.step()
    net_o=net(X_test)
    pred = net_o.data.max(1)[1]
    
    pres,recall,f_1,_=precision_recall_fscore_support(y_test, pred, average=None)
    print(pres,recall,f_1)
    
    p.append(pres[0])
    r.append(recall[0])
    f.append(f_1[0])
    
    p1.append(pres[1])
    r1.append(recall[1])
    f1.append(f_1[1])
    
    p2.append(pres[2])
    r2.append(recall[2])
    f2.append(f_1[2])
    
    p3.append(pres[3])
    r3.append(recall[3])
    f3.append(f_1[3])

plt.plot(range(epochs),np.array(LOSS),label='Train')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xscale('log')
plt.legend()
plt.title('NN Learning')
plt.savefig('loss.png')
    
    
test_loss=0    
#y_pred=net(X_test)
net_o=net(X_test)
#test_loss += criterion(net_out, y_test).detach().
pred = net_o.data.max(1)[1]  # get the index of the mhttp://localhost:8888/notebooks/Documents/Tesis/CNN%20qso/Reportes%20Proyecto%20de%20Monografia/Noteboooks/Weekly%20Reports.ipynb#ax log-probability
print(pred)
"""


# In[ ]:


plt.plot(range(epochs),np.array(p),label='Stars',c='r')
plt.plot(range(epochs),np.array(p1),label='Galaxies',c='g')
plt.plot(range(epochs),np.array(p2),label='QSO',c='b')
plt.plot(range(epochs),np.array(p3),label='QSO_BAL',c='silver')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision')
plt.legend()
plt.savefig('precision.png')


# In[ ]:


plt.plot(range(epochs),np.array(r),label='Stars',c='r')
plt.plot(range(epochs),np.array(r1),label='Galaxies',c='g')
plt.plot(range(epochs),np.array(r2),label='QSO',c='b')
plt.plot(range(epochs),np.array(r3),label='QSO_BAL',c='silver')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall')
plt.legend()
plt.savefig('recall.png')


# In[ ]:


plt.plot(np.array(r),np.array(p),label='Stars',c='r')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Stars')
plt.legend()

plt.savefig('prs.png')


# In[ ]:


plt.plot(np.array(r1),np.array(p1),label='Galaxies',c='g')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Galaxy Precision-Recall ')
plt.legend()

plt.savefig('prg.png')


# In[ ]:


plt.plot(np.array(r2),np.array(p2),label='QSO',c='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('QSO Precision-Recall ')
plt.legend()

plt.savefig('prq.png')


# In[ ]:


plt.plot(np.array(r3),np.array(p3),label='QSO_BAL',c='silver')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('QSO_BAL Precision-Recall ')
plt.legend()

plt.savefig('prqb.png')


# In[ ]:


plt.plot(range(epochs),np.array(f),label='Stars',c='r')
plt.plot(range(epochs),np.array(f1),label='Galaxies',c='g')
plt.plot(range(epochs),np.array(f2),label='QSO',c='b')
plt.plot(range(epochs),np.array(f3),label='QSO_BAL',c='silver')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.title('F1')
plt.legend()
plt.savefig('F1.png')


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
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
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
plot_confusion_matrix(y_test, pred, classes=class_names, title='Confusion matrix')
plt.savefig('cm_test.png')
from sklearn.metrics import precision_recall_fscore_support
prf=precision_recall_fscore_support(y_test, pred, average=None)#,labels=['Star','Galaxy','QSO','QSO_BAL'])

print(prf)


# In[ ]:


ip = torch.randn(1, 2, 2)
print(ip[0,0], ip[0].shape)
#for i in enumerate(ip,0):
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




