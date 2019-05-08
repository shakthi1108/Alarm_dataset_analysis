import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from datetime import date
import calendar
#import pandas_profilin

time_cols = ['ALARM_TIME','CANCEL_TIME','ACK_TIME','EVENT_TIME','TERMINATED_TIME', 'DELETED_BY']
df = pd.read_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/fm_alarms_18012019_085951.csv"
#                 , parse_dates=time_cols
                 )
    
cols = ['INSERT_TIME','DELETED_BY','INTERNAL_ALARM','AGENT_ID_TYPE','INSERT_TIME','TROUBLE_TICKET_ID','CANCEL_TIME','ACK_TIME','ACK_STATUS','NOTIFICATION_ID','FILTERED_FROM','TEXT','EVENT_TIME','ALARM_STATUS','TROUBLE_TICKET_ID','INTERNAL_ALARM','USER_ADDITIONAL_INFO','SUPPLEMENTARY_INFO','DIAGNOSTIC_INFO','EXTRA_TEXT','ADDITIONAL_TEXT5','ADDITIONAL_TEXT6','ADDITIONAL_TEXT7','NE_GID','NE_CONT_GID','CORRELATED_ALARM','CORRELATING_ALARM','UPDATE_TIMESTAMP','CORRELATOR_CREATED','PIPE_KEY','NOTES_INDICATOR','ORIGINAL_DN','LIFTED_DN','ALARM_UPLOAD_SYNC','ADAPTATION_ID','ADAPTATION_RELEASE','OC_ID']
    
df.drop(cols, axis=1, inplace=True)
    
df['ALARM_TIME'] = pd.to_datetime(df['ALARM_TIME'], dayfirst=True)
#df['CANCEL_TIME'] = pd.to_datetime(df['CANCEL_TIME'], dayfirst=True)
#df['ACK_TIME'] = pd.to_datetime(df['ACK_TIME'], dayfirst=True)
#df['EVENT_TIME'] = pd.to_datetime(df['EVENT_TIME'], dayfirst=True)
    
    
df = df.iloc[0 : 6952]
df['TERMINATED_TIME'] = pd.to_datetime(df['TERMINATED_TIME'], dayfirst=True)
    
    
df=df[ df['TERMINATED_TIME'].notnull() & df['ALARM_TIME'].notnull() ]
df['diff_seconds'] = df['TERMINATED_TIME'] - df['ALARM_TIME']
df = df[df.diff_seconds > '0 days 00:10:00']

df['ALARM_TIME'] = pd.to_datetime(df['ALARM_TIME'])
include = df[df['ALARM_TIME'].dt.year == 2018]
df = df[df['ALARM_TIME'].dt.year != 2018]

df['Day of Week'] = df['ALARM_TIME'].dt.weekday_name
    
#df.convert_objects(convert_numeric=True)
    
#def handle_non_numerical_data(df):
##    columns = df.columns.values
#     columns = ['DN', 'CANCELLED_BY', 'ACKED_BY']
#        
#     for column in columns:
#            text_digit_vals = {}
#            def convert_to_int(val):
#                return text_digit_vals[val]
#            
#            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
#                column_contents = df[column].values.tolist()
#                unique_elements = set(column_contents)
#                x=0
#                for unique in unique_elements:
#                    if unique not in text_digit_vals:
#                        text_digit_vals[unique] = x
#                        x += 1
#                        
#                df[column] = list(map(convert_to_int, df[column]))
#                
#     return df
#    
#df = handle_non_numerical_data(df)
#    

f = plt.figure(1)

df = df.sort_values('ALARM_TIME')
plt.plot_date(df.ALARM_TIME, df.AGENT_ID, color = 'k')
plt.legend()
plt.show()


f.show()

i = plt.figure(2)
    
#plt.plot_date(df.ALARM_TIME, df.DN, color = 'r')

#
df = df.sort_values('ALARM_TIME')
df = df.sort_values('TERMINATED_TIME')
plt.plot_date(df.ALARM_TIME, df.AGENT_ID, color = 'k')
plt.plot_date(df.TERMINATED_TIME, df.AGENT_ID, color = 'r')
plt.legend()
plt.show()

#ax = plt.subplot2grid((1,1),(0,0))
#for label in ax.xaxis.get_ticklabels():
#        label.set_rotation(45)
##    #
#df = df.sort_values('DN')
#plt.plot(df.DN, df.ALARM_TIME,'-', linewidth = 5.0)
##df_new1_test = df_new1_test.sort_values('ALARM_NUMBER')
##plt.plot(df_new1_test.ALARM_NUMBER, df_new1_test.ACK_TIME, '-', linewidth = 2.5, color = 'k')
##df_new1_test = df_new1_test.sort_values('ALARM_NUMBER')
##plt.plot(df_new1_test.ALARM_NUMBER, df_new1_test.CANCEL_TIME, '-', linewidth = 1.0)
#plt.legend()
#plt.show()
#plt.title('DN')
i.show()
    
g = plt.figure(3)

from_ts = '2019-01-09 08:30:00'
to_ts = '2019-01-09 08:50:00'
df1 = df[(df['ALARM_TIME'] < from_ts) | (df['ALARM_TIME'] > to_ts)]

from_ts1 = '2019-01-03 18:30:00'
to_ts1 = '2019-01-03 18:35:00'
df1 = df1[(df1['ALARM_TIME'] < from_ts) | (df1['ALARM_TIME'] > to_ts)]

df1 = df1.sort_values('Day of Week')
plt.scatter(df1['Day of Week'], df1.AGENT_ID, color = 'k')

plt.legend()
plt.show()
g.show()

df1['Hour'] = df1.ALARM_TIME.dt.hour

l = plt.figure(4)
plt.scatter(df1['Hour'], df1.AGENT_ID, color = 'k')

plt.legend()
plt.show()
l.show()


##START PLOTTING GRAPH WITH AGENT ID INSTEAD OF DN!!!!