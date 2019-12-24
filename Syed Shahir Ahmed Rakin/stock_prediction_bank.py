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

#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 20,10

n_reservoir= 600
sparsity= 0.4
rand_seed= 20
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

train_len = 1100
future = 10
futureTotal= 30
pred_tot_abbank=np.zeros(futureTotal)
pred_tot_bankasia=np.zeros(futureTotal)
pred_tot_ebl=np.zeros(futureTotal)
pred_tot_dutchbangl=np.zeros(futureTotal)
pred_tot_ific=np.zeros(futureTotal)
pred_tot_jamunabank=np.zeros(futureTotal)
pred_tot_mtb=np.zeros(futureTotal)
pred_tot_nccbank=np.zeros(futureTotal)
pred_tot_southeast=np.zeros(futureTotal)
pred_tot_ucb=np.zeros(futureTotal)

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfabbank_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_abbank[i:i+future] = prediction[:,0]

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfbankasia_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_bankasia[i:i+future] = prediction[:,0]
    
for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfebl_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_ebl[i:i+future] = prediction[:,0]

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfdutchbangl_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_dutchbangl[i:i+future] = prediction[:,0]

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfific_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_ific[i:i+future] = prediction[:,0]

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfjamunabank_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_jamunabank[i:i+future] = prediction[:,0]
  
for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfmtb_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_mtb[i:i+future] = prediction[:,0]

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfnccbank_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_nccbank[i:i+future] = prediction[:,0]

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfsoutheast_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_southeast[i:i+future] = prediction[:,0]

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(train_len),dfucb_N[i:train_len+i])
    prediction = esn.predict(np.ones(future))
    pred_tot_ucb[i:i+future] = prediction[:,0]
  
    
#Graph Section Starts Here
    
fig,axs1 = plt.subplots(2)
fig.subplots_adjust(hspace=.5)

axs1[0].plot(range(00,train_len+futureTotal),dfabbank_N[00:train_len+futureTotal],'b',label="Original Data (AB Bank)", alpha=0.3)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs1[0].plot(range(train_len,train_len+futureTotal),pred_tot_abbank,'k',  alpha=0.8, label='Post ESN Data (AB Bank)')

axs1[1].plot(range(00,train_len+futureTotal),dfbankasia_N[00:train_len+futureTotal],'y',label="Original Data (Bank Asia)", alpha=0.3)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs1[1].plot(range(train_len,train_len+futureTotal),pred_tot_bankasia,'r',  alpha=0.8, label='Post ESN Data (Bank Asia)')
axs1[0].plot([train_len,train_len],[dfabbank_N.min(),dfabbank_N.max()],'b--', linewidth=2)
axs1[1].plot([train_len,train_len],[dfbankasia_N.min(),dfbankasia_N.max()],'r--', linewidth=2)

axs1[0].set_title('Ground Truth and Echo State Network Output', fontsize=12)
axs1[1].set_title('Ground Truth and Echo State Network Output', fontsize=12)

for ax in axs1.flat:
    ax.set(xlabel='Time (Days)', ylabel='Price ($)')
    
axs1[0].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
axs1[1].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
plt.show()
###################################

fig,axs2 = plt.subplots(2)
fig.subplots_adjust(hspace=.5)

axs2[0].plot(range(00,train_len+futureTotal),dfific_N[00:train_len+futureTotal],'b',label="Original Data (IFIC)", alpha=0.4)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs2[0].plot(range(train_len,train_len+futureTotal),pred_tot_ific,'k',  alpha=0.8, label='Post ESN Data (IFIC)')

axs2[1].plot(range(00,train_len+futureTotal),dfjamunabank_N[00:train_len+futureTotal],'y',label="Original Data (Jamuna Bank)", alpha=0.7)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs2[1].plot(range(train_len,train_len+futureTotal),pred_tot_jamunabank,'r',  alpha=0.8, label='Post ESN Data (Jamuna Bank)')
axs2[0].plot([train_len,train_len],[dfific_N.min(),dfific_N.max()],'b--', linewidth=2)
axs2[1].plot([train_len,train_len],[dfjamunabank_N.min(),dfjamunabank_N.max()],'r--', linewidth=2)

axs2[0].set_title('Ground Truth and Echo State Network Output', fontsize=12)
axs2[1].set_title('Ground Truth and Echo State Network Output', fontsize=12)

for ax in axs2.flat:
    ax.set(xlabel='Time (Days)', ylabel='Price ($)')
    
axs2[0].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
axs2[1].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
plt.show()

##################################

fig,axs2 = plt.subplots(2)
fig.subplots_adjust(hspace=.5)

axs2[0].plot(range(00,train_len+futureTotal),dfmtb_N[00:train_len+futureTotal],'b',label="Original Data (MTB)", alpha=0.4)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs2[0].plot(range(train_len,train_len+futureTotal),pred_tot_mtb,'k',  alpha=0.8, label='Post ESN Data (MTB)')

axs2[1].plot(range(00,train_len+futureTotal),dfnccbank_N[00:train_len+futureTotal],'y',label="Original Data (NCC Bank)", alpha=0.7)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs2[1].plot(range(train_len,train_len+futureTotal),pred_tot_nccbank,'r',  alpha=0.8, label='Post ESN Data (NCC Bank)')
axs2[0].plot([train_len,train_len],[dfmtb_N.min(),dfmtb_N.max()],'b--', linewidth=2)
axs2[1].plot([train_len,train_len],[dfnccbank_N.min(),dfnccbank_N.max()],'r--', linewidth=2)

axs2[0].set_title('Ground Truth and Echo State Network Output', fontsize=12)
axs2[1].set_title('Ground Truth and Echo State Network Output', fontsize=12)

for ax in axs2.flat:
    ax.set(xlabel='Time (Days)', ylabel='Price ($)')
    
axs2[0].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
axs2[1].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
plt.show()

##################################

fig,axs2 = plt.subplots(2)
fig.subplots_adjust(hspace=.5)

axs2[0].plot(range(00,train_len+futureTotal),dfsoutheast_N[00:train_len+futureTotal],'b',label="Original Data (South East Bank)", alpha=0.4)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs2[0].plot(range(train_len,train_len+futureTotal),pred_tot_southeast,'k',  alpha=0.8, label='Post ESN Data (South East Bank)')

axs2[1].plot(range(00,train_len+futureTotal),dfucb_N[00:train_len+futureTotal],'y',label="Original Data (UCB)", alpha=0.7)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs2[1].plot(range(train_len,train_len+futureTotal),pred_tot_ucb,'r',  alpha=0.8, label='Post ESN Data (UCB)')
lo,hi = plt.ylim()
axs2[0].plot([train_len,train_len],[dfsoutheast_N.min(),dfsoutheast_N.max()],'b--', linewidth=2)
axs2[1].plot([train_len,train_len],[dfucb_N.min(),dfucb_N.max()],'r--', linewidth=2)

axs2[0].set_title('Share Status of South East Bank (Used ESN)', fontsize=12)
axs2[1].set_title('Share Status of UCB (Used ESN)', fontsize=12)

for ax in axs2.flat:
    ax.set(xlabel='Time (Days)', ylabel='Price ($)')
    
axs2[0].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
axs2[1].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
plt.show()

##################################

fig,axs2 = plt.subplots(2)
fig.subplots_adjust(hspace=.5)

axs2[0].plot(range(00,train_len+futureTotal),dfebl_N[00:train_len+futureTotal],'b',label="Original Data (EBL)", alpha=0.4)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs2[0].plot(range(train_len,train_len+futureTotal),pred_tot_ebl,'k',  alpha=0.8, label='Post ESN Data (EBL)')

axs2[1].plot(range(00,train_len+futureTotal),dfdutchbangl_N[00:train_len+futureTotal],'y',label="Original Data (DBBL)", alpha=0.7)
#plt.plot(range(0,train_len),pred_training,'.g',  alpha=0.3)
axs2[1].plot(range(train_len,train_len+futureTotal),pred_tot_dutchbangl,'r',  alpha=0.8, label='Post ESN Data (DBBL)')
axs2[0].plot([train_len,train_len],[dfebl_N.min(),dfebl_N.max()],'b--', linewidth=2)
axs2[1].plot([train_len,train_len],[dfdutchbangl_N.min(),dfdutchbangl_N.max()],'r--', linewidth=2)

axs2[0].set_title('Share Status of EBL (Used ESN)', fontsize=12)
axs2[1].set_title('Share Status of DBBL (Used ESN)', fontsize=12)

for ax in axs2.flat:
    ax.set(xlabel='Time (Days)', ylabel='Price ($)')
    
axs2[0].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
axs2[1].legend(bbox_to_anchor=(0.80, 1.2), fontsize='small', loc=2, borderaxespad=0.)
plt.show()

sns.despine()