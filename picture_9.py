import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font1 = {
'weight' : 'normal',
'size'   : 12,
}

#BLE
data1=[]
data1=pd.read_excel("./excel/1.4acc_data1_BLE.xlsx",header=None)
denominator=len(data1[0])#分母数量
Data1=pd.Series(data1[0])#将数据转换为Series利用分组频数计算
Fre1=Data1.value_counts()
Fre1_sort=Fre1.sort_index(axis=0,ascending=True)
Fre1_df=Fre1_sort.reset_index()#将Series数据转换为DataFrame
Fre1_df[0]=Fre1_df[0]/denominator#转换成概率
Fre1_df.columns=['Rds','Fre1']
Fre1_df['cumsum']=np.cumsum(Fre1_df['Fre1'])

data2=[]
data2=pd.read_excel("./excel/11.4acc_data1_BLE.xlsx",header=None)
denominator=len(data2[0])#分母数量
Data2=pd.Series(data2[0])#将数据转换为Series利用分组频数计算
Fre2=Data2.value_counts()
Fre2_sort=Fre2.sort_index(axis=0,ascending=True)
Fre2_df=Fre2_sort.reset_index()#将Series数据转换为DataFrame
Fre2_df[0]=Fre2_df[0]/denominator#转换成概率
Fre2_df.columns=['Rds','Fre2']
Fre2_df['cumsum']=np.cumsum(Fre2_df['Fre2'])



#Capo

data4=[]
data4=pd.read_excel("./excel/5acc_data2_Capo.xlsx",header=None)
denominator=len(data4[0])#分母数量
Data4=pd.Series(data4[0])#将数据转换为Series利用分组频数计算
Fre4=Data4.value_counts()
Fre4_sort=Fre4.sort_index(axis=0,ascending=True)
Fre4_df=Fre4_sort.reset_index()#将Series数据转换为DataFrame
Fre4_df[0]=Fre4_df[0]/denominator#转换成概率
Fre4_df.columns=['Rds','Fre4']
Fre4_df['cumsum']=np.cumsum(Fre4_df['Fre4'])

data5=[]
data5=pd.read_excel("./excel/11.4acc_data2_Capo.xlsx",header=None)
denominator=len(data5[0])#分母数量
Data5=pd.Series(data5[0])#将数据转换为Series利用分组频数计算
Fre5=Data5.value_counts()
Fre5_sort=Fre5.sort_index(axis=0,ascending=True)
Fre5_df=Fre5_sort.reset_index()#将Series数据转换为DataFrame
Fre5_df[0]=Fre5_df[0]/denominator#转换成概率
Fre5_df.columns=['Rds','Fre5']
Fre5_df['cumsum']=np.cumsum(Fre5_df['Fre5'])

# plot=plt.figure(figsize=(9, 4))
#
# ax1=plot.add_subplot(1,2,1)
# ax1.plot(Fre1_df['Rds'],Fre1_df['cumsum'], linestyle='--', color='#6699ff',label='Original')
# ax1.plot(Fre4_df['Rds'],Fre4_df['cumsum'],color='#00cc33',label='Capo')
# ax1.set_title("$\gamma$ = 1.4")
# ax1.set_xlabel("Accuracy", font1)
# ax1.set_ylabel("Cumulative Probability", font1)
# ax1.set_xlim(0.75,1)
# plt.legend(fontsize = 12, loc = 'upper left',framealpha=0)
#
# ax1=plot.add_subplot(1,2,2)
# ax1.plot(Fre2_df['Rds'],Fre2_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
# ax1.plot(Fre5_df['Rds'],Fre5_df['cumsum'],color='#00cc33',label='Capo')
# ax1.set_title("$\gamma$ = 11.4")
# ax1.set_xlabel("Accuracy", font1)
# ax1.set_ylabel("Cumulative Probability", font1)
# ax1.set_xlim(0.75,1)
# plt.legend(fontsize = 12, loc = 'upper left',framealpha=0)


a=plt.figure(figsize=(8, 6))

plt.plot(Fre1_df['Rds'],Fre1_df['cumsum'], linestyle='--', color='#6699ff',label='Original_gamma_1.4')
plt.plot(Fre2_df['Rds'],Fre2_df['cumsum'],linestyle='--', color='#00cc33',label='Original_gamma_11.4')
plt.plot(Fre4_df['Rds'],Fre4_df['cumsum'],color='#6699ff',label='Capo_gamma_1.4')
plt.plot(Fre5_df['Rds'],Fre5_df['cumsum'],color='#00cc33',label='Capo_gamma_11.4')
plt.xlabel("Accuracy", font1)
plt.ylabel("Cumulative Probability", font1)
plt.xlim(0.75,1)
plt.legend(fontsize = 16, loc = 'upper left',framealpha=0)



plt.savefig(r'CDFacc.pdf', dpi=300, bbox_inches = 'tight',  format='pdf')
plt.show()


