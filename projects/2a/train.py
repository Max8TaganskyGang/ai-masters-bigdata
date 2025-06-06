import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

from model import model, fields

logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

try:
  proj_id = sys.argv[1]
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")

read_table_opts = dict(sep="\t", names=fields, na_values='\\N')
df = pd.read_table(train_path, **read_table_opts)

X_train, X_test, y_train, y_test = train_test_split(
df.iloc[:, 2:], df.iloc[:, 1], test_size=0.33, random_state=42
)


model.fit(X_train, y_train)

model_score = model.score(X_test, y_test)

logging.info(f"model score: {model_score:.3f}")

# save the model
dump(model, "{}.joblib".format(proj_id))
