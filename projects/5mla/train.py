import os
import sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)]
fields = ["id", "label"] + numeric_features + categorical_features

def main(train_path, model_param1=0.1):
   
    with mlflow.start_run():
        mlflow.log_param("C", model_param1)
        
    
        read_table_opts = dict(sep="\t", names=fields)
        df = pd.read_table(train_path, **read_table_opts)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df.iloc[:, 2:], df.iloc[:, 1], test_size=0.33, random_state=42
        )
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('logreg', LogisticRegression(C=model_param1, max_iter=1024))
        ])
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_proba = model.predict_proba(X_test)
        test_log_loss = log_loss(y_test, y_proba)
        

        mlflow.log_metric("log_loss", test_log_loss)
        
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train_path = sys.argv[1]
    model_param1 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    main(train_path, model_param1)