import pandas as pd
import numpy as np 
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

infile = sys.argv[1]

data = pd.read_csv(infile)

### segregate the dataframe into features and target

#x = data.iloc[:, :-1]
#y = data.iloc[:, -1:]

x = data.drop(['activity'],axis = 1).values
y = data['activity'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)


from sklearn.ensemble import RandomForestClassifier
res = []
for i in range(20,120,20):
    classifier = RandomForestClassifier(n_estimators=i)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy_test = accuracy_score(y_test,y_pred)
    print(accuracy_test)

