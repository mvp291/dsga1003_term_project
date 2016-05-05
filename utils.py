import numpy as np
import pandas as pd

def separate_target(train, val, test, target):
    X_train, y_train = train.ix[:, :-3], train.ix[:, target]
    X_val, y_val = val.ix[:, :-3], val.ix[:, target]
    X_test, y_test = test.ix[:, :-3], test.ix[:, target]
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_pred_to_file(y_test, y_pred, test, filename='results.csv'):
    ''' Build dataframe to compare real values with predictions on test set '''
    new_df = pd.concat([y_test, pd.Series(y_pred, index=y_test.index), test.ix[:, -3:]], axis=1)
    new_df.to_csv(filename)


def loss_azimuth(y_truth, y_predicted):
    dist_upper = abs(y_truth - y_predicted)
    dist_lower = 360 - abs(y_truth - y_predicted)
    az_loss = np.minimum(dist_upper, dist_lower)
    return az_loss.mean()


def loss_elevation(y_truth,y_predicted):
    el_loss = abs(y_truth - y_predicted)
    return el_loss.mean()
