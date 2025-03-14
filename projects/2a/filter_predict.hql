ADD FILE projects/2a/model.py;
ADD FILE projects/2a/predict.py;
ADD FILE 2a.joblib;

INSERT INTO TABLE hw2_pred 
SELECT TRANSFORM(*)
USING '/opt/conda/envs/dsenv/bin/python3 predict.py' AS (id, pred)
FROM hw2_test WHERE if1 > 20 AND if1 < 40;
