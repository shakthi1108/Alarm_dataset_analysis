import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date
from datetime import datetime
import calendar


if len(sys.argv) == 7:
    location = sys.argv[1]
    severity = int(sys.argv[2])
    topcount = int(sys.argv[3])
    timediff = int(sys.argv[4])
    Dayofweek = sys.argv[5]
    Hourofday = int(sys.argv[6])
    
else:
    print('\n Please run the program with : \n\n Argument 1 : Path of file \n Argument 2 : Severity \n Argument 3 : Top n most frequent occured data \n Argument 4: Time diff filter in double quotes \n')
    quit()
 
#location = sys.argv[1]
#severity = int(sys.argv[2])
#topcount = int(sys.argv[3])
#timediff = str(sys.argv[4])    
    
    
time_cols = ['ALARM_TIME','CANCEL_TIME','ACK_TIME','EVENT_TIME','TERMINATED_TIME', 'DELETED_BY']
df = pd.read_csv(location)
    
cols = ['INSERT_TIME','DELETED_BY','INTERNAL_ALARM','AGENT_ID_TYPE','INSERT_TIME','TROUBLE_TICKET_ID','CANCEL_TIME','ACK_TIME','ACK_STATUS','NOTIFICATION_ID','FILTERED_FROM','TEXT','EVENT_TIME','ALARM_STATUS','TROUBLE_TICKET_ID','INTERNAL_ALARM','USER_ADDITIONAL_INFO','SUPPLEMENTARY_INFO','DIAGNOSTIC_INFO','EXTRA_TEXT','ADDITIONAL_TEXT5','ADDITIONAL_TEXT6','ADDITIONAL_TEXT7','NE_GID','NE_CONT_GID','CORRELATED_ALARM','CORRELATING_ALARM','UPDATE_TIMESTAMP','CORRELATOR_CREATED','PIPE_KEY','NOTES_INDICATOR','ORIGINAL_DN','LIFTED_DN','ALARM_UPLOAD_SYNC','ADAPTATION_ID','ADAPTATION_RELEASE','OC_ID']
    
df.drop(cols, axis=1, inplace=True)

df['ALARM_TIME'] = pd.to_datetime(df['ALARM_TIME'], dayfirst=True)
df['Day of Week'] = df['ALARM_TIME'].dt.weekday_name
df['Hour'] = df.ALARM_TIME.dt.hour
df = df.iloc[0 : 6952]
df['TERMINATED_TIME'] = pd.to_datetime(df['TERMINATED_TIME'], dayfirst=True)
    
    
df=df[ df['TERMINATED_TIME'].notnull() & df['ALARM_TIME'].notnull() ]
df['diff_seconds'] = df['TERMINATED_TIME'] - df['ALARM_TIME']
df['diff_seconds']=df['diff_seconds']/np.timedelta64(1,'m')
df = df[df.diff_seconds > 1]
df['NE'] = df['DN'].str.split("/").str[1].str.strip("',]")

i = plt.figure(1)    
df = df.sort_values('ALARM_TIME')
df = df.sort_values('TERMINATED_TIME')
plt.plot_date(df.ALARM_TIME, df.NE, color = 'k')
plt.plot_date(df.TERMINATED_TIME, df.NE, color = 'r')
plt.legend()
plt.show()  
i.show()  

df = df[df.SEVERITY == severity]

df = df[df.diff_seconds > timediff]

df = df[df['Day of Week'] == Dayofweek]
df = df[df['Hour'] == Hourofday]


print(df['NE'].value_counts().head(topcount))


m = plt.figure(2)
df = df.sort_values('ALARM_TIME')
df = df.sort_values('TERMINATED_TIME')
plt.plot_date(df.ALARM_TIME, df.NE, color = 'k')
plt.plot_date(df.TERMINATED_TIME, df.AGENT_ID, color = 'r')
plt.legend()
plt.show()
m.show()


#g = plt.figure(2)
#
#df = df.sort_values('Day of Week')
#plt.scatter(df1['Day of Week'], df1.AGENT_ID, color = 'k')
#
#plt.legend()
#plt.show()
#g.show()
#
#
#l = plt.figure(3)
#plt.scatter(df1['Hour'], df1.AGENT_ID, color = 'k')
#
#plt.legend()
#plt.show()
#l.show()




