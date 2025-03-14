#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import fields

logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

# Загрузка модели
model = load("2a.joblib")

# Удаление целевой переменной из списка полей
fields.remove("label")

# Опции для чтения данных
read_opts = dict(
    sep='\t', names=fields, index_col=False, header=None,
    iterator=True, chunksize=10000, na_values='\\N'
)

# Чтение данных из stdin
for df in pd.read_csv(sys.stdin, **read_opts):
    # Предсказание вероятностей
    pred_proba = model.predict_proba(df)[:, 1]
    
    # Формирование результата
    for row_id, prob in zip(df.id, pred_proba):
        print(f"{row_id}\t{prob:.6f}")
