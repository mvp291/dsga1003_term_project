import numpy as np
import pandas as pd


def loss_azimuth(y_truth, y_predicted):
    dist_upper = abs(y_truth - y_predicted)
    dist_lower = 360 - abs(y_truth - y_predicted)
    az_loss = np.minimum(dist_upper, dist_lower)
    return az_loss.mean()


def loss_elevation(y_truth,y_predicted):
    el_loss = abs(y_truth - y_predicted)
    return el_loss.mean()
