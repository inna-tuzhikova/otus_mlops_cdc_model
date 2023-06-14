import os
import random

import mlflow
from mlflow import log_artifacts, log_param, log_metric


mlflow.set_tracking_uri('http://51.250.22.177:5000/')
mlflow.set_experiment('mlflow_test')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://51.250.22.177:9000'


def main():
    log_param('param_1', random.randint(0, 100))

    log_metric('m1', random.random() + 1)
    log_metric('m2', random.random() + 2)
    log_metric('m3', random.random() + 3)

    folder, filename = 'artifacts', 'artifact.txt'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename), 'w') as f:
        f.write('hello world')
    log_artifacts(folder)


if __name__ == '__main__':
    main()
