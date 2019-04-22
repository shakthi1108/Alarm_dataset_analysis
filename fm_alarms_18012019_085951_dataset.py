import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas_profiling
#import time
#import psutil

#time_cols = ['ALARM_TIME','CANCEL_TIME','ACK_TIME','EVENT_TIME']
df = pd.read_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/fm_alarms_18012019_085951.csv"
#                 , parse_dates=time_cols
                 )
#print(df.head())

cols = ['INSERT_TIME', 'AGENT_ID', 'OC_ID','ADAPTATION_RELEASE','TERMINATED_TIME', 'TROUBLE_TICKET_ID','ADAPTATION_ID','NOTES_INDICATOR','PIPE_KEY','INTERNAL_ALARM','USER_ADDITIONAL_INFO','SUPPLEMENTARY_INFO','DIAGNOSTIC_INFO','EXTRA_TEXT','ADDITIONAL_TEXT5','ADDITIONAL_TEXT6','ADDITIONAL_TEXT7','NE_GID','NE_CONT_GID','CORRELATED_ALARM','CORRELATING_ALARM','UPDATE_TIMESTAMP','CORRELATOR_CREATED','PIPE_KEY','NOTES_INDICATOR','ORIGINAL_DN','LIFTED_DN','ALARM_UPLOAD_SYNC','ADAPTATION_ID','ADAPTATION_RELEASE','OC_ID']
df.drop(cols, axis=1, inplace=True)

df['ALARM_TIME'] = pd.to_datetime(df['ALARM_TIME'], dayfirst=True)
df['CANCEL_TIME'] = pd.to_datetime(df['CANCEL_TIME'], dayfirst=True)
df['ACK_TIME'] = pd.to_datetime(df['ACK_TIME'], dayfirst=True)
df['EVENT_TIME'] = pd.to_datetime(df['EVENT_TIME'], dayfirst=True)



df.to_csv('C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/fm_alarms_18012019_085951_edited.csv')

print(type(df.ALARM_TIME[0]))


#
#df.convert_objects(convert_numeric=True)
#
#def handle_non_numerical_data(df):
##    columns = df.columns.values
#    columns = ['DN','CANCELLED_BY','ACKED_BY','TEXT','AGENT_ID']
#    
#    for column in columns:
#        text_digit_vals = {}
#        def convert_to_int(val):
#            return text_digit_vals[val]
#        
#        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
#            column_contents = df[column].values.tolist()
#            unique_elements = set(column_contents)
#            x=0
#            for unique in unique_elements:
#                if unique not in text_digit_vals:
#                    text_digit_vals[unique] = x
#                    x += 1
#                    
#            df[column] = list(map(convert_to_int, df[column]))
#            
#    return df
#
#df = handle_non_numerical_data(df)

#print(df.dtypes)

stat = df.describe()

#profile = pandas_profiling.ProfileReport(df)
#display(profile)
#profile.to_file(outputfile="C:/Users/shakgane/Downloads/alarmDump/myoutputfile.html")


#correlations=df.corr()
#fig=plt.figure()
#ax=fig.add_subplot(111)
#cax=ax.matshow(correlations,vmin=-1,vmax=1)
#fig.colorbar(cax)

#plt.scatter(df['DN'],df['AGENT_ID'])

#df['AGENT_ID'] = df['AGENT_ID'].apply(lambda v:(v-df['AGENT_ID'].min())/(df['AGENT_ID'].max()-df['AGENT_ID'].min()))
#df['DN'] = df['DN'].apply(lambda v:(v-df['DN'].min())/(df['DN'].max()-df['DN'].min()))


#linreg = LinearRegression()
#
#p1 = np.polyfit(df['AGENT_ID'], df['DN'], 1)
#plt.plot(df['AGENT_ID'], df['DN'], 'o', color='r')
#plt.plot(df['AGENT_ID'], np.polyval(p1,df['AGENT_ID']), 'k-')
#
#plt.show()

#x_train, x_test, y_train, y_test = train_test_split(df['AGENT_ID'].values.reshape(-1,1), df['DN'].values.reshape(-1,1),random_state=1)
#linreg.fit(x_train, y_train)
#ypred = linreg.predict(x_test)
#print(r2_score(y_test, ypred))


#pd.plotting.scatter_matrix(df)
