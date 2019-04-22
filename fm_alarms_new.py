import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas_profiling

time_cols = ['ALARM_TIME','CANCEL_TIME','ACK_TIME','EVENT_TIME','TERMINATED_TIME', 'DELETED_BY']
df = pd.read_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/fm_alarms_18012019_085951.csv"
#                 , parse_dates=time_cols
                 )

cols = ['INSERT_TIME','TROUBLE_TICKET_ID','INTERNAL_ALARM','USER_ADDITIONAL_INFO','SUPPLEMENTARY_INFO','DIAGNOSTIC_INFO','EXTRA_TEXT','ADDITIONAL_TEXT5','ADDITIONAL_TEXT6','ADDITIONAL_TEXT7','NE_GID','NE_CONT_GID','CORRELATED_ALARM','CORRELATING_ALARM','UPDATE_TIMESTAMP','CORRELATOR_CREATED','PIPE_KEY','NOTES_INDICATOR','ORIGINAL_DN','LIFTED_DN','ALARM_UPLOAD_SYNC','ADAPTATION_ID','ADAPTATION_RELEASE','OC_ID']

df.drop(cols, axis=1, inplace=True)

df['ALARM_TIME'] = pd.to_datetime(df['ALARM_TIME'], dayfirst=True)
df['CANCEL_TIME'] = pd.to_datetime(df['CANCEL_TIME'], dayfirst=True)
df['ACK_TIME'] = pd.to_datetime(df['ACK_TIME'], dayfirst=True)
df['EVENT_TIME'] = pd.to_datetime(df['EVENT_TIME'], dayfirst=True)

df=df[ df['CANCEL_TIME'].notnull() & df['ALARM_TIME'].notnull() ]
df['diff_seconds'] = df['CANCEL_TIME'] - df['ALARM_TIME']
df['diff_seconds']=df['diff_seconds']/np.timedelta64(1,'s')

df = df[df.diff_seconds>1000]

df.to_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/fm_alarms_18012019_085951_consolidated.csv")

df1 = pd.read_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/fm_alarms_18012019_085951_consolidated.csv"
                 , parse_dates=time_cols
                 )

profile = pandas_profiling.ProfileReport(df1)
display(profile)
profile.to_file(outputfile="C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/consolidated.html")

