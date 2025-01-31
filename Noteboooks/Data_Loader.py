# Jairo Andres Saavedra Alfonso
# 01/09/2019
# Universidad de los Andes
# Phycis 
######################__________________Data Loader 1.0__________________######################

# Data Loader for SpectraNET


#Packages
from astropy.io import fits
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import torch 
import torch.nn as nn
import fitsio 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from torch.autograd import Variable
import torch.utils.data
import time
import os
from astropy.modeling import models
import astropy.units as u
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import find_lines_derivative
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import gaussian_smooth
from specutils.analysis import gaussian_sigma_width, gaussian_fwhm, fwhm, fwzi,line_flux,snr_derived,equivalent_width




def Load_Files(file_1,file_2,N_sample,objts,classification=False):
    print('INFO:')
    #hdul = fitsio.FITS(file_1) # Open file 1 -- 'truth_DR12Q.fits'
    #info=hdul.info() # File info
    hdul = fits.open(file_1,mode='denywrite') 
    #data=hdul[1].read() # Database of spectra with human-expert classifications
    data=hdul[1].data    
    #print('The file {} have {} objects. \n'.format(file_1,data.shape[0]))     

    print('INFO:')
    # Reading data from data_dr12.fits. This file had the spectra from data dr12. 
    #hdul_2 = fitsio.FITS(file_2) # Open file 2 -- 'data_dr12.fits'
    #info2=hdul_2.info() # File info
    #data2=hdul_2[1].read() # Database of spectra
    #spectra=hdul_2[0].read() # Spectrum of each object
    hdul_2 = fits.open(file_2,mode='denywrite') 
    data2=hdul_2[1].data # Database of spectra
    spectra=hdul_2[0].data # Spectrum of each object

    #print('The file {} have {} spectra. \n'.format(file_2,spectra.shape[0]))
 
    	
    # Subset of PLATE parameters of both data
    data_PLATE_1=data['PLATE']
    data_PLATE_2=data2['PLATE']

    # Subset of MJD parameters of both data
    data_MJD_1=data['MJD']
    data_MJD_2=data2['MJD']

    # Subset of FIBERID parameters of both data
    data_FIBERID_1=data['FIBERID']
    data_FIBERID_2=data2['FIBERID']
    data_ID_1=data['THING_ID']
    data_ID_2=data2['TARGETID']
    
    objts=np.asarray(objts)
    
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

    print('INFO: There is available')
    print('-->Star:',STAR.shape[0])
    print('-->Galaxy:',GALAXY.shape[0])
    print('-->QSO:',QSO.shape[0])
    print('-->QSO BAL:',QSO_BAL.shape[0])
    print('-->NN: {}\n'.format(N_C.shape[0]))
    
    # I create two DataFrame for Superset_DR12Q and data_dr12 with only three parameters
    data1={'PLATE':data_PLATE_1,'MJD':data_MJD_1,'FIBERID':data_FIBERID_1,'ID':data_ID_1}
    data1=pd.DataFrame(data=data1)

    data2={'PLATE':data_PLATE_2,'MJD':data_MJD_2,'FIBERID':data_FIBERID_2,'ID':data_ID_2}
    data2=pd.DataFrame(data=data2)

    # I convert all objects in both set to string chain in orden to combine them as one new ID.
    data1['PLATE']=data1['PLATE'].astype(str)
    data1['MJD']=data1['MJD'].astype(str)
    data1['FIBERID']=data1['FIBERID'].astype(str)
    data1['PM'] = data1['MJD'].str.cat(data1['FIBERID'],sep="-")
    data1['NEWID'] = data1['PLATE'].str.cat(data1['PM'],sep="-")
    data_1=data1.drop(columns=['PLATE','MJD','FIBERID','ID','PM']).values

    data2['PLATE']=data2['PLATE'].astype(str)
    data2['MJD']=data2['MJD'].astype(str)
    data2['FIBERID']=data2['FIBERID'].astype(str)
    data2['PM'] = data2['MJD'].str.cat(data2['FIBERID'],sep="-")
    data2['NEWID'] = data2['PLATE'].str.cat(data2['PM'],sep="-")
    data_2=data2.drop(columns=['PLATE','MJD','FIBERID','ID','PM']).values # New set of database 2 with new ID's

    # With the routine of numpy intersect1d, I find the intersections elements in both sets. This elements  
    data_CO=np.array(np.intersect1d(data_1,data_2,return_indices=True))

    data_CO_objects=data_CO[0] # The unique new ID of each element in both sets
    data_CO_ind1=data_CO[1] # Indices of intersected elements from the original data 1 (Superset_DR12Q.fits) 
    data_CO_ind2=data_CO[2] # Indices of intersected elements form the original data 2 (data_dr12.fits)
    print('INFO:')
    print('I find {} objects with spectra from DR12 \n'.format(len(data_CO_objects)))
    indi={'ind1':data_CO_ind1,'ind2':data_CO_ind2}
    ind=pd.DataFrame(data=indi,index=data_CO_ind1)

    cp=np.array(data['CLASS_PERSON'],dtype=float)
    z=np.array(data['Z_VI'],dtype=float)
    zc=np.array(data['Z_CONF_PERSON'],dtype=float)
    bal=np.array(data['BAL_FLAG_VI'],dtype=float)
    bi=np.array(data['BI_CIV'],dtype=float)

    d={'CLASS_PERSON':cp,'Z_VI':z,'Z_CONF_PERSON':zc,'BAL_FLAG_VI':bal,'BI_CIV':bi}
    data_0=pd.DataFrame(data=d)
    
    obj=data_0.loc[data_CO_ind1]
    
    if(classification!=True):

        if(objts[0]=='QSO'):
            qsos=obj.loc[obj['CLASS_PERSON']==3]
            qsos=qsos.loc[qsos['Z_CONF_PERSON']==3]
            sample_objects=qsos.sample(n=int(N_sample),weights='CLASS_PERSON', random_state=5)
            
            indi=np.array(sample_objects.index)
            indi1=ind.loc[indi].values
            
        elif(objts[0]=='QSO_BAL'):
            qsos_bal=obj.loc[obj['CLASS_PERSON']==30]
            qsos_bal=qsos_bal.loc[qsos_bal['Z_CONF_PERSON']==3]
            sample_objects=qsos_bal.sample(n=int(N_sample),weights='CLASS_PERSON', random_state=5)
            
            indi=np.array(sample_objects.index)
            indi1=ind.loc[indi].values
            
        elif(len(objts)==2):
            qsos=obj.loc[obj['CLASS_PERSON']==3]
            qsos=qsos.loc[qsos['Z_CONF_PERSON']==3]
            
            qsos_bal=obj.loc[obj['CLASS_PERSON']==30] 
            qsos_bal=qsos_bal.loc[qsos_bal['Z_CONF_PERSON']==3]
            
            sample_qso=qsos.sample(n=int(N_sample/2),weights='CLASS_PERSON', random_state=5)
            sample_qso_bal=qsos_bal.sample(n=int(N_sample/2),weights='CLASS_PERSON', random_state=5)
            sample_objects=pd.concat([sample_qso,sample_qso_bal])
            
            ind_qso=np.array(sample_qso.index)
            ind_qso_bal=np.array(sample_qso_bal.index)
            
            indi=np.concatenate((ind_qso,ind_qso_bal), axis=None)
            indi1=ind.loc[indi].values

        spectra_=np.zeros((N_sample,886))
            
        j=0
        kernel_size=5
        flux_threshold=1.1
        parameters=np.zeros((N_sample,7)) #Number of lines // FHWM of max emission line // EW of max emission line // Spectrum Mean // Spectrum STDV // Spectrum Flux Integral // Spectrum SNR  
        for i in indi:
            k=indi1[j,1]
            x=np.linspace(3600,10500,443)
            zero_spectrum=spectra[k,:443]
            spectrum = Spectrum1D(flux=zero_spectrum*u.Jy, spectral_axis=x*u.AA)
            
            #Continuum fit and gaussian smooth
            g1_fit = fit_generic_continuum(spectrum)
            y_continuum_fitted = g1_fit(x*u.AA)
            spec_nw_2=spectrum/y_continuum_fitted
            spectrum_smooth = gaussian_smooth(spec_nw_2,kernel_size)

            #Number of lines
            lines_1 = find_lines_derivative(spectrum_smooth, flux_threshold=flux_threshold)
            l=lines_1[lines_1['line_type']=='emission']
            number_lines=l['line_center_index'].shape[0]
            parameters[j,0]=number_lines

            #FWHM
            parameters[j,1]=fwhm(spectrum_smooth).value

            #EW
            parameters[j,2]=equivalent_width(spectrum_smooth).value

            #Spectrum Mean
            parameters[j,3]=np.mean(spectrum_smooth.flux)

            #Spectrum STDV
            parameters[j,4]=np.std(spectrum_smooth.flux)

            #Spectrum Flux Integral
            parameters[j,5]=line_flux(spectrum_smooth).value

            #Spectrum SNR
            parameters[j,6]=snr_derived(spectrum_smooth).value
            j+=1  

        d={'Lines_Number':parameters[:,0],'FHWM':parameters[:,1],'EW':parameters[:,2],'Mean':parameters[:,3],'STDV':parameters[:,4],'STDV':parameters[:,4],'Spectrum_Flux':parameters[:,5],'SNR':parameters[:,6]}

        parameters=pd.DataFrame(data=d)
        #X=spectra_.values
        
        #mean_flx= np.ma.average(X[:,:443],axis=1)
        #ll=(X[:,:443]-mean_flx.reshape(-1,1))**2
        #aveflux=np.ma.average(ll, axis=1)
        #sflux = np.sqrt(aveflux)
        #X = (X[:,:443]-mean_flx.reshape(-1,1))/sflux.reshape(-1,1)        
        
        y=sample_objects['Z_VI']
        y=np.array(y,dtype=float)
        #y_max=np.max(y)
        #y=y/y_max
        
        return parameters,y
            
    stars=obj.loc[obj['CLASS_PERSON']==1]
    galaxies=obj.loc[obj['CLASS_PERSON']==4]
    qsos=obj.loc[obj['CLASS_PERSON']==3]
    qsos_bal=obj.loc[obj['CLASS_PERSON']==30]

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

    spectra_=np.zeros((N_sample,886))
    j=0
    for i in indi:
        k=indi1[j,1]
        spectra_[j,:]=spectra[k,:]
        j=j+1    
        
        
    spectra_=pd.DataFrame(spectra_)
    X=spectra_.values
    
    #Renormalize spectra
    
    mean_flx= np.ma.average(X[:,:443], axis=1)
    ll=(X[:,:443]-mean_flx.reshape(-1,1))**2
    aveflux=np.ma.average(ll,axis=1)
    sflux = np.sqrt(aveflux)
    X = (X[:,:443]-mean_flx.reshape(-1,1))/sflux.reshape(-1,1)

    y=sample_objects['CLASS_PERSON']
    y=y.replace([1, 4, 3, 30], [0,1,2,3]).values
    y=np.array(y,dtype=float)

    return X,y


def Data_Loader(X, y, N_sample, batch_size, test_size, val_size,classification=True):

    n_train=int(N_sample*(1-test_size)*(1-val_size))
    n_test=int(N_sample*(test_size))
    n_val=int((n_train-n_test)*(val_size))
    batch_size_test=int(n_test/100)
    batch_size_val=int(n_test/100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=1)
    
    train_data = []
    
    for i in range(y_train.shape[0]):
        xt=X_train[i,:].reshape(1,-1)
        if(classification==True):
            train_data.append([Variable(torch.tensor(xt, dtype=torch.float)), torch.tensor(y_train[i], dtype=torch.long)])
        #else:
            #train_data.append([Variable(torch.tensor(xt, dtype=torch.float)), torch.tensor(y_train[i], dtype=torch.float)])
    
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

    test_data = []
    for i in range(y_test.shape[0]):
        xtst=X_test[i,:].reshape(1,-1)
        if(classification==True):
            test_data.append([Variable(torch.tensor(xtst, dtype=torch.float)), torch.tensor(y_test[i], dtype=torch.long)])
        #else:
            #test_data.append([Variable(torch.tensor(xtst, dtype=torch.float)), torch.tensor(y_test[i], dtype=torch.float)])
    
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size_test)
    
    val_data = []
    for i in range(y_val.shape[0]):
        xv=X_val[i,:].reshape(1,-1)
        if(classification==True):
            val_data.append([Variable(torch.tensor(xv, dtype=torch.float)), torch.tensor(y_val[i], dtype=torch.long)])
        #else:
            #val_data.append([Variable(torch.tensor(xv, dtype=torch.float)), torch.tensor(y_val[i], dtype=torch.float)])
    
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=batch_size_val)

    train_s=y_train.shape[0]
    test_s=y_test.shape[0]
    val_s=y_val.shape[0]
    
    print('INFO:')
    print('Data Train: {}'.format(y_train.shape[0]))
    print('Data Validation: {}'.format(y_val.shape[0]))
    print('Data Test: {}'.format(y_test.shape[0]))
    return train_loader,test_loader,val_loader,train_s,test_s,val_s

