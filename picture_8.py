import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font1 = {
'weight' : 'normal',
'size'   : 12,
}

#BLE
data1=[]
data1=pd.read_excel("./excel/5acc_data1_BLE.xlsx",header=None)
denominator=len(data1[0])#分母数量
Data1=pd.Series(data1[0])#将数据转换为Series利用分组频数计算
Fre1=Data1.value_counts()
Fre1_sort=Fre1.sort_index(axis=0,ascending=True)
Fre1_df=Fre1_sort.reset_index()#将Series数据转换为DataFrame
Fre1_df[0]=Fre1_df[0]/denominator#转换成概率
Fre1_df.columns=['Rds','Fre1']
Fre1_df['cumsum']=np.cumsum(Fre1_df['Fre1'])

data2=[]
data2=pd.read_excel("./excel/5pre_data1_BLE.xlsx",header=None)
denominator=len(data2[0])#分母数量
Data2=pd.Series(data2[0])#将数据转换为Series利用分组频数计算
Fre2=Data2.value_counts()
Fre2_sort=Fre2.sort_index(axis=0,ascending=True)
Fre2_df=Fre2_sort.reset_index()#将Series数据转换为DataFrame
Fre2_df[0]=Fre2_df[0]/denominator#转换成概率
Fre2_df.columns=['Rds','Fre2']
Fre2_df['cumsum']=np.cumsum(Fre2_df['Fre2'])

data3=[]
data3=pd.read_excel("./excel/5rec_data1_BLE.xlsx",header=None)
denominator=len(data3[0])#分母数量
Data3=pd.Series(data3[0])#将数据转换为Series利用分组频数计算
Fre3=Data3.value_counts()
Fre3_sort=Fre3.sort_index(axis=0,ascending=True)
Fre3_df=Fre3_sort.reset_index()#将Series数据转换为DataFrame
Fre3_df[0]=Fre3_df[0]/denominator#转换成概率
Fre3_df.columns=['Rds','Fre3']
Fre3_df['cumsum']=np.cumsum(Fre3_df['Fre3'])

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
data5=pd.read_excel("./excel/5pre_data2_Capo.xlsx",header=None)
denominator=len(data5[0])#分母数量
Data5=pd.Series(data5[0])#将数据转换为Series利用分组频数计算
Fre5=Data5.value_counts()
Fre5_sort=Fre5.sort_index(axis=0,ascending=True)
Fre5_df=Fre5_sort.reset_index()#将Series数据转换为DataFrame
Fre5_df[0]=Fre5_df[0]/denominator#转换成概率
Fre5_df.columns=['Rds','Fre5']
Fre5_df['cumsum']=np.cumsum(Fre5_df['Fre5'])


dat6=[]
dat6=pd.read_excel("./excel/5rec_data2_Capo.xlsx",header=None)
denominator=len(dat6[0])#分母数量
Dat6=pd.Series(dat6[0])#将数据转换为Series利用分组频数计算
Fre6=Dat6.value_counts()
Fre6_sort=Fre6.sort_index(axis=0,ascending=True)
Fre6_df=Fre6_sort.reset_index()#将Series数据转换为DataFrame
Fre6_df[0]=Fre6_df[0]/denominator#转换成概率
Fre6_df.columns=['Rds','Fre6']
Fre6_df['cumsum']=np.cumsum(Fre6_df['Fre6'])


# plot=plt.figure(figsize=(14, 4))
#
# ax1=plot.add_subplot(1,3,1)
# ax1.plot(Fre1_df['Rds'],Fre1_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
# ax1.plot(Fre4_df['Rds'],Fre4_df['cumsum'],color='#00cc33',label='Capo')
# ax1.plot([0,0.6],[1,0.6],color = 'red',linestyle='--')
# #ax1.set_title("CDF")
# ax1.set_xlabel("Accuracy", font1)
# ax1.set_ylabel("Cumulative Probability", font1)
# ax1.set_xlim(0.8,1)
# plt.legend(fontsize = 12, loc = 'upper left',framealpha=0)
#
# ax1=plot.add_subplot(1,3,2)
# ax1.plot(Fre2_df['Rds'],Fre2_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
# ax1.plot(Fre5_df['Rds'],Fre5_df['cumsum'],color='#00cc33',label='Capo')
# #ax1.set_title("CDF")
# ax1.set_xlabel("Precision", font1)
# ax1.set_ylabel("Cumulative Probability", font1)
# ax1.set_xlim(0,1)
# plt.legend(fontsize = 12, loc = 'upper left',framealpha=0)
#
#
# ax1=plot.add_subplot(1,3,3)
# ax1.plot(Fre3_df['Rds'],Fre3_df['cumsum'],linestyle='--', color='#6699ff',label='Original')
# ax1.plot(Fre6_df['Rds'],Fre6_df['cumsum'],color='#00cc33',label='Capo')
# #ax1.set_title("CDF")
# ax1.set_xlabel("Recall", font1)
# ax1.set_ylabel("Cumulative Probability", font1)
# ax1.set_xlim(0,1)
# plt.legend(fontsize = 12, loc = 'upper left',framealpha=0)

a=plt.figure(figsize=(8, 6))

plt.plot(Fre1_df['Rds'],Fre1_df['cumsum'],linestyle='--', color='#6699ff',label='Original_Accuracy')
plt.plot(Fre2_df['Rds'],Fre2_df['cumsum'],linestyle='--', color='#00cc33',label='Original_Precision')
plt.plot(Fre3_df['Rds'],Fre3_df['cumsum'],linestyle='--', color='#DC0126',label='Original_Recall')
plt.plot(Fre4_df['Rds'],Fre4_df['cumsum'],color='#6699ff',label='Capo_Accuracy')
plt.plot(Fre5_df['Rds'],Fre5_df['cumsum'],color='#00cc33',label='Capo_Precision')
plt.plot(Fre6_df['Rds'],Fre6_df['cumsum'],color='#DC0126',label='Capo_Recall')
plt.xlabel("Value", font1)
plt.ylabel("Cumulative Probability", font1)
plt.xlim(0,1)
plt.legend(fontsize = 16, loc = 'upper left',framealpha=0)


plt.savefig(r'CDF5.pdf', dpi=300, bbox_inches = 'tight',  format='pdf')
plt.show()


