#!/opt/conda/envs/dsenv/bin/python

import os, sys
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

path_in, path_out = sys.argv[1], sys.argv[2]


df = pd.read_csv(path_in)

X_train, X_test, y_train, y_test = train_test_split(
     df.iloc[:, 1:], df.iloc[:, 0], test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model_score = model.score(X_test, y_test)

dump(model, path_out)