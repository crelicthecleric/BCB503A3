import argparse
from numpy import mean
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

parser = argparse.ArgumentParser(description='validate classifications')
parser.add_argument('--kfold', action=argparse.BooleanOptionalAction, help="do k-fold validation")
parser.add_argument('input', help='input file')
parser.add_argument('output', help='output file')
parser.add_argument('model', help='SVM or RF')
parser.add_argument('treeqty', type=int, nargs='?', default=100, help='Number of trees for random forest')

args = parser.parse_args()

def validate_model(doKfold, input, output, model, numTrees):
    frac = float(re.findall(r"[0-9][.][0-9]+", input)[-1])
    data = pd.read_csv(input)
    X = data.drop(columns=["activity"])
    X = StandardScaler().fit_transform(X)
    y = data["activity"]
    
    if model == "SVM":
        clf = SVC()
        name = model
    elif model == "RF":
        clf = RandomForestClassifier(n_estimators=numTrees, random_state=1)
        name =  "RF: " + str(numTrees) + " Trees"
        
    if doKfold:
        cv = StratifiedKFold(random_state=1, shuffle=True)
        score = mean(cross_val_score(clf, X, y, scoring='f1_macro', cv=cv, n_jobs=-1))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = f1_score(y_test, y_pred, average='macro')
        
    entry = [frac, score, name]
    result = pd.DataFrame([entry], columns=["fraction", "score", "type"])
    result.to_csv(output, index=False)

if __name__ == "__main__":
    validate_model(args.kfold, args.input, args.output, args.model, args.treeqty)