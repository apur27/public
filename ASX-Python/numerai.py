# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:55:04 2019

@author: UPuroAb
"""

import pandas as pd
import sklearn.linear_model
#import numerapi

# download the latest training dataset (takes around 30s)
training_data = pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz")
training_data.head()
# download the latest tournament dataset (takes around 30s)
tournament_data = pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz")
tournament_data.head()
# find only the feature columns
feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
# select those columns out of the training dataset
training_features = training_data[feature_cols]
# create a model and fit the training data (~30 sec to run)
model = sklearn.linear_model.LinearRegression()
model.fit(training_features, training_data.target_kazutsugi)



# select the feature columns from the tournament data
live_features = tournament_data[feature_cols]
# predict the target on the live features
predictions = model.predict(live_features)
# predictions must have an `id` column and a `prediction_kazutsugi` column
predictions_df = tournament_data["id"].to_frame()
predictions_df["prediction_kazutsugi"] = predictions
predictions_df.head()

## Get your API keys from https://numer.ai/submit
#public_id = "REPLACEME"
#secret_key = "REPLACEME"
#napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
#
## Upload your predictions
#predictions_df.to_csv("predictions.csv", index=False)
#submission_id = napi.upload_predictions("predictions.csv")