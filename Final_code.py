import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from datetime import date
from datetime import datetime
import calendar

#reading table
time_cols = ['ALARM_TIME','CANCEL_TIME','ACK_TIME','EVENT_TIME','TERMINATED_TIME', 'DELETED_BY']
df = pd.read_csv("C:/Users/shakgane/Downloads/alarmDump/a4ossv04_fm_dump.tar/fm_alarms_18012019_085951.csv"
#                 , parse_dates=time_cols
                 )

#dropping irrelevant columns    
cols = ['INSERT_TIME','DELETED_BY','INTERNAL_ALARM','AGENT_ID_TYPE','INSERT_TIME','TROUBLE_TICKET_ID','CANCEL_TIME','ACK_TIME','ACK_STATUS','NOTIFICATION_ID','FILTERED_FROM','TEXT','EVENT_TIME','ALARM_STATUS','TROUBLE_TICKET_ID','INTERNAL_ALARM','USER_ADDITIONAL_INFO','SUPPLEMENTARY_INFO','DIAGNOSTIC_INFO','EXTRA_TEXT','ADDITIONAL_TEXT5','ADDITIONAL_TEXT6','ADDITIONAL_TEXT7','NE_GID','NE_CONT_GID','CORRELATED_ALARM','CORRELATING_ALARM','UPDATE_TIMESTAMP','CORRELATOR_CREATED','PIPE_KEY','NOTES_INDICATOR','ORIGINAL_DN','LIFTED_DN','ALARM_UPLOAD_SYNC','ADAPTATION_ID','ADAPTATION_RELEASE','OC_ID']
df.drop(cols, axis=1, inplace=True)
    
#changing format of timestamp
df['ALARM_TIME'] = pd.to_datetime(df['ALARM_TIME'], dayfirst=True)
df = df.iloc[0 : 6952]
df['TERMINATED_TIME'] = pd.to_datetime(df['TERMINATED_TIME'], dayfirst=True)

#Finding out time duration of alarm
df=df[ df['TERMINATED_TIME'].notnull() & df['ALARM_TIME'].notnull() ]
df['diff_seconds'] = df['TERMINATED_TIME'] - df['ALARM_TIME']
df['diff_seconds']=df['diff_seconds']/np.timedelta64(1,'m')

#removing severity = 4
df = df[df.SEVERITY<4]

#extracting Network element
df['NE'] = df['DN'].str.split("/").str[1].str.strip("',]")
#
#counts = {}
#for each in df.NE:
#    count = {}
#    print (each + ':')
#    for row in columns[each]:
#        count[row] = count.get(row,0) + 1
#    counts[each] = count

#converting to numerical data
df.convert_objects(convert_numeric=True)
    
def handle_non_numerical_data(df):
#    columns = df.columns.values
     columns = ['NE']
        
     for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]
            
            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x=0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x += 1
                        
                df['NE1'] = list(map(convert_to_int, df[column]))
                
     return df
    
df = handle_non_numerical_data(df)

first = df['NE'].value_counts().head(1)
second = df['NE'].value_counts().head(2)


alarms_by_topn = df.groupby( [ "NE"] )["NE"].size().sort_values(ascending = False).head(5)
alarm_by_NE = alarms_by_topn.to_frame(name = 'count').reset_index()

for index, row in alarm_by_NE.iterrows():
        #    print(row['NE'], row['count'])
        print (f"\nNE {row['NE']} has {row['count']} alarm(/s)\n")
#        if row['count'] > window: # The rolling window should have those many inputs as well to calcuate the values
            # But this could loose out if the number of days the alarms generated attribute to an anomaly but the days less than the window size
        alarm_by_NE_new = df.loc[df.NE == row['NE']]
        alarm_by_NE_new = alarm_by_NE_new.groupby( [ "ALARM_TIME"] ).size().to_frame(name = 'count').reset_index()




#df.sort_values(by='NE', inplace=True)

q1, q2, q3 = np.percentile(df['NE1'],[25,50,75])

IQR = q3-q1
lower_boundary = q1 - 1.5 * IQR
upper_boundary = q3 + 1.5 * IQR

print("\n 25th percentile is : " + str(q1))
print(df['NE'][df['NE1'] == q1])

print("\n 50th percentile is : " + str(q2))
print(df['NE'][df['NE1'] == q2])

print("\n 75th percentile is : " + str(q3))
print(df['NE'][df['NE1'] == q3])

plt.boxplot(df.NE1)


