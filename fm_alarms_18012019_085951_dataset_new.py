import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
#import pandas_profiling


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



#df_new = df[df.DN=='PLMN-CBIS/CBIS-PCRFSPS1-NOR/COMPUTE-overcloud-ovscompute-23/SERVER-1']

#df_new.to_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/DN__PLMN_CBISCBIS_PCRFSPS1_NORCOMPUTE_overcloud_ovscompute_23SERVER_1/fm_alarms_18012019_085951_new.csv")

df_new1 = df[df.DN=='PLMN-CBIS/CBIS-PCRFSPS1-NOR/COMPUTE-overcloud-ovscompute-23/SERVER-1']
df_new1.to_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/DN__PLMN_PLMNNAFD_6/fm_alarms_18012019_085951_new1.csv")

df_new1_test = pd.read_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/DN__PLMN_PLMNNAFD_6/fm_alarms_18012019_085951_new1.csv", parse_dates = time_cols)


stat = df_new1.describe()

df_new1_test=df_new1_test[ df_new1_test['CANCEL_TIME'].notnull() & df_new1_test['ALARM_TIME'].notnull() ]
df_new1_test['diff_seconds'] = df_new1_test['CANCEL_TIME'] - df_new1_test['ALARM_TIME']
df_new1_test['diff_seconds']=df_new1_test['diff_seconds']/np.timedelta64(1,'s')
df_new1_test.to_csv('C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/DN__PLMN_PLMNNAFD_6/fm_alarms_18012019_085951_new1.csv')


#profile = pandas_profiling.ProfileReport(df_new1)
#display(profile)
#profile.to_file(outputfile="C:/Users/shakgane/Downloads/alarmDump/myoutputfile_new1.html")

#plt.scatter(df_new1['SEVERITY'], df_new1['ALARM_NUMBER'])
#plt.title('severity distribtion')
#plt.xlabel('severity')
#plt.ylabel('Alarm Number')
#plt.show()



#i = plt.figure(1)

#plt.plot_date(df_new1.ALARM_TIME, df_new1.SEVERITY, color = 'r')
#
#df_new1 = df_new1.sort_values('CANCEL_TIME')
#plt.plot_date(df_new1.CANCEL_TIME, df_new1.SEVERITY, color = 'k')
#plt.legend()
#plt.show()

#for label in ax.xaxis.get_ticklabels():
#    label.set_rotation(45)
#
#df_new1_test = df_new1_test.sort_values('ALARM_NUMBER')
#plt.plot(df_new1_test.ALARM_NUMBER, df_new1_test.ALARM_TIME,'-', linewidth = 5.0)
#df_new1_test = df_new1_test.sort_values('ALARM_NUMBER')
#plt.plot(df_new1_test.ALARM_NUMBER, df_new1_test.ACK_TIME, '-', linewidth = 2.5, color = 'k')
#df_new1_test = df_new1_test.sort_values('ALARM_NUMBER')
#plt.plot(df_new1_test.ALARM_NUMBER, df_new1_test.CANCEL_TIME, '-', linewidth = 1.0)
#plt.legend()
#plt.show()
#plt.title('Alarm Number')
#i.show()
#
#g = plt.figure(2)
#ax = plt.subplot2grid((1,1),(0,0))
# 
#for label in ax.xaxis.get_ticklabels():
#    label.set_rotation(45)
#    
#df_new1_test = df_new1_test.sort_values('SEVERITY')
#plt.plot(df_new1_test.SEVERITY, df_new1_test.ALARM_TIME,'-', linewidth = 5.0)
#df_new1_test = df_new1_test.sort_values('SEVERITY')
#plt.plot(df_new1_test.SEVERITY, df_new1_test.CANCEL_TIME, '-', linewidth = 2.0)
#plt.legend()
#plt.title('Severity')
#plt.show()
#
#g.show()
#
#f = plt.figure(3)
#ax = plt.subplot2grid((1,1),(0,0))
# 
#for label in ax.xaxis.get_ticklabels():
#    label.set_rotation(45)
#    
#df_new1_test = df_new1_test.sort_values('ALARM_TYPE')
#plt.plot(df_new1_test.ALARM_TYPE, df_new1_test.ALARM_TIME,'-', linewidth = 5.0)
#df_new1_test = df_new1_test.sort_values('ALARM_TYPE')
#plt.plot(df_new1_test.ALARM_TYPE, df_new1_test.CANCEL_TIME, '-', linewidth = 2.0)
#plt.legend()
#plt.show()
#plt.title('alarm type')
#
#f.show()


