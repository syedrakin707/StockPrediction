# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:14:15 2019

@author: Admin
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pyESN import ESN

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

n_reservoir=500
sparsity=0.4
rand_seed=15
spectral_radius = 1.2
noise = .0010


esn = ESN(n_inputs = 1,
      n_outputs = 1, 
      n_reservoir = n_reservoir,
      sparsity=sparsity,
      random_state=rand_seed,
      spectral_radius = spectral_radius,
      noise=noise)

df = pd.read_csv("bank_stock_prediction.csv")

dfabbank = df['CLOSEP (ABBANK)']
dfabbank_N=dfabbank.to_numpy()

dfbankasia = df['CLOSEP (BANKASIA)']
dfbankasia_N=dfbankasia.to_numpy()

dfebl = df['CLOSEP (EBL)']
dfebl_N=dfebl.to_numpy()

dfdutchbangl = df['CLOSEP (DUTCHBANGL)']
dfdutchbangl_N=dfdutchbangl.to_numpy()

dfific = df['CLOSEP (IFIC)']
dfific_N=dfific.to_numpy()

dfjamunabank = df['CLOSEP (JAMUNABANK)']
dfjamunabank_N=dfjamunabank.to_numpy()

dfmtb = df['CLOSEP (MTB)']
dfmtb_N=dfmtb.to_numpy()

dfnccbank = df['CLOSEP (NCCBANK)']
dfnccbank_N=dfnccbank.to_numpy()

dfsoutheast = df['CLOSEP (SOUTHEASTB)']
dfsoutheast_N=dfsoutheast.to_numpy()

dfucb = df['CLOSEP (UCB)']
dfucb_N=dfbankasia.to_numpy()
print(dfabbank_N)
#for i in range(0,len(df)):
#     dfabbank['CLOSEP (ABBANK)'][i] = df['CLOSEP (ABBANK)'][i]

train_len = 1000
future = 10
futureTotal=50
pred_tot_abbank=np.zeros(futureTotal)
pred_tot_bankasia=np.zeros(futureTotal)

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfabbank_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_abbank[i:i+future] = prediction[:,0]

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfbankasia_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_bankasia[i:i+future] = prediction[:,0]
    
plt.figure(figsize=(16,8))
fig1,axs = plt.subplots(2)
fig1.subplots_adjust(hspace=.5)

axs[0].plot(range(00,train_len+futureTotal),dfabbank[00:train_len+futureTotal],'b',label="Original Data (AB Bank)", alpha=0.3)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs[0].plot(range(train_len,train_len+futureTotal),pred_tot_abbank,'k',  alpha=0.8, label='Post ESN Data (AB Bank)')

axs[1].plot(range(00,train_len+futureTotal),dfbankasia[00:train_len+futureTotal],'y',label="Original Data (Bank Asia)", alpha=0.3)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs[1].plot(range(train_len,train_len+futureTotal),pred_tot_bankasia,'r',  alpha=0.8, label='Post ESN Data (Bank Asia)')
lo,hi = plt.ylim()
axs[0].plot([train_len,train_len],[lo+np.spacing(1),hi-np.spacing(1)],'b--', linewidth=4)
axs[1].plot([train_len,train_len],[lo+np.spacing(1),hi-np.spacing(1)],'r--', linewidth=4)

axs[0].set_title('Ground Truth and Echo State Network Output', fontsize=12)
axs[1].set_title('Ground Truth and Echo State Network Output', fontsize=12)

for ax in axs.flat:
    ax.set(xlabel='Time (Days)', ylabel='Price ($)')
    
axs[0].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
axs[1].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)

sns.despine()
#print(df.head())
#print(df.axes)
#print(df.values)

#print(df.shape)