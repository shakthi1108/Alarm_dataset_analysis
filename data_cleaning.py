import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

df = pd.read_csv("C:/Users/shakgane/Downloads/titanic.csv")
#print(df.head())


#df.fillna(0, inplace=True)
#df.drop([1309], inplace=True)

#enc = LabelEncoder()
#enc.fit(df['sex'])
#df['sex'] = enc.transform(df['sex'])

#stupid way of converting non numeric data into numeric
df.convert_objects(convert_numeric=True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    
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
                    
            df[column] = list(map(convert_to_int, df[column]))
            
    return df

df = handle_non_numerical_data(df)
#print(df.tail())

x = np.array(df.drop(['survived'], 1).astype(float))
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(x)

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
        
print(correct/len(x))
    
    
    
    
    