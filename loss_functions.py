import numpy as np

def loss_azimuth(y_truth,y_predicted):
    dist_upper = abs(y_truth-y_predicted)
    dist_lower = 360 - abs(y_truth-y_predicted)
    az_loss = np.minimum(dist_upper,dist_lower)
    return az_loss


def loss_elevation(y_truth,y_predicted):
    el_loss = abs(y_truth-y_predicted)
    return el_loss