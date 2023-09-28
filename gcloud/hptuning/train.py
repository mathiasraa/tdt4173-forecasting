# %%
import argparse
import datetime
import os
import pandas as pd
import subprocess
import pickle

from google.cloud import storage
import hypertune
import xgboost as xgb
from random import shuffle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-dir",  # handled automatically by AI Platform
    help="GCS location to write checkpoints and export models",
    required=True,
)
parser.add_argument(
    "--max_depth",  # Specified in the config file
    help="Maximum depth of the XGBoost tree. default: 3",
    default=3,
    type=int,
)
parser.add_argument(
    "--n_estimators",  # Specified in the config file
    help="Number of estimators to be created. default: 100",
    default=100,
    type=int,
)
parser.add_argument(
    "--min_child_weight",  # Specified in the config file
    help="which booster to use: gbtree, gblinear or dart. default: gbtree",
    default=3,
    type=int,
)
parser.add_argument(
    "--colsample_bytree",  # Specified in the config file
    help="which booster to use: gbtree, gblinear or dart. default: gbtree",
    default=0.7,
    type=float,
)
parser.add_argument(
    "--subsample",  # Specified in the config file
    help="which booster to use: gbtree, gblinear or dart. default: gbtree",
    default=0.8,
    type=float,
)
parser.add_argument(
    "--reg_alpha",  # Specified in the config file
    help="which booster to use: gbtree, gblinear or dart. default: gbtree",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--gamma",  # Specified in the config file
    help="which booster to use: gbtree, gblinear or dart. default: gbtree",
    default=0.2,
    type=float,
)
parser.add_argument(
    "--reg_lambda",  # Specified in the config file
    help="which booster to use: gbtree, gblinear or dart. default: gbtree",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--eta",  # Specified in the config file
    help="which booster to use: gbtree, gblinear or dart. default: gbtree",
    default=0.1,
    type=float,
)


args = parser.parse_args()


# ---------------------------------------
# This is where your model code would go. Below is an example model using the auto mpg dataset.
# ---------------------------------------
# Define the format of your input data including unused columns
# (These are the columns from the auto-mpg data files)

COLUMNS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model-year",
    "origin",
    "car-name",
]

FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model-year",
    "origin",
]

TARGET = "y"

bucket = storage.Client().bucket("neuralnet-ninjas")
# Path to the data inside the public bucket
X_train_blob = bucket.blob("forecasting_data/X_train.csv")
y_train_blob = bucket.blob("forecasting_data/y_train.csv")
X_test_blob = bucket.blob("forecasting_data/X_test.csv")
y_test_blob = bucket.blob("forecasting_data/y_test.csv")

# Download the data
X_train_blob.download_to_filename("X_train.csv")
y_train_blob.download_to_filename("y_train.csv")
X_test_blob.download_to_filename("X_test.csv")
y_test_blob.download_to_filename("y_test.csv")

X_train = pd.read_csv("./X_train.csv", index_col="time")
y_train = pd.read_csv("./y_train.csv", index_col="time")
X_test = pd.read_csv("./X_test.csv", index_col="time")
y_test = pd.read_csv("./y_test.csv", index_col="time")

# Create the regressor, here we will use a Lasso Regressor to demonstrate the use of HP Tuning.
# Here is where we set the variables used during HP Tuning from
# the parameters passed into the python script


regressor = xgb.XGBRegressor(
    max_depth=args.max_depth,
    n_estimators=args.n_estimators,
    min_child_weight=args.min_child_weight,
    colsample_bytree=args.colsample_bytree,
    subsample=args.subsample,
    reg_alpha=args.reg_alpha,
    gamma=args.gamma,
    reg_lambda=args.reg_lambda,
    eta=args.eta,
)

# Transform the features and fit them to the regressor
regressor.fit(X_train, y_train)


# Calculate the mean accuracy on the given test data and labels.
score = regressor.score(X_test, y_test)

# The default name of the metric is training/hptuning/metric.
# We recommend that you assign a custom name. The only functional difference is that
# if you use a custom name, you must set the hyperparameterMetricTag value in the
# HyperparameterSpec object in your job request to match your chosen name.
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#HyperparameterSpec
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="my_metric_tag", metric_value=score, global_step=1000
)

# Export the model to a file
model_filename = "model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(regressor, f)

# Example: job_dir = 'gs://BUCKET_ID/xgboost_job_dir/1'
job_dir = args.job_dir.replace("gs://", "")  # Remove the 'gs://'
# Get the Bucket Id
bucket_id = job_dir.split("/")[0]
# Get the path
bucket_path = job_dir[len("{}/".format(bucket_id)) :]  # Example: 'xgboost_job_dir/1'

# Upload the model to GCS
bucket = storage.Client().bucket(bucket_id)
blob = bucket.blob("{}/{}".format(bucket_path, model_filename))

blob.upload_from_filename(model_filename)
