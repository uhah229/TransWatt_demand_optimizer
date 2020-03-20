#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf


def create_predictions(X, y, time_steps=1):
    Xs, ys = [], []
    v = X.iloc[0:(0 + time_steps)].values
    Xs.append(v)
    ys.append(y.iloc[0 + time_steps])
    return np.array(Xs), np.array(ys)


def make_n_predictions(Q,model,n=6,time_steps=24):
    """
        Function to generate n predictions based on Q
        time_steps is the window to move (depends on LSTM time_step training)
        model is the trained LSTM model
    """
    last_Q = Q.copy()
    for i in range(n):
        last_Q = last_Q[-time_steps-2:-1]
        X_predict, Q_predict = create_predictions(last_Q, last_Q.energy, time_steps)
        predict_next = model.predict(X_predict)
        last_Q = last_Q.append({"energy": predict_next[0][0]}, ignore_index=True)
    return last_Q[-n:]

def get_price_from_Q(Q):
    Q0 = 30
    # Q30 = 55.4213045642437
    # Q70 = 77.81349172281733
    Q30 = 45
    Q70 = 70
    if Q < Q0:
        return 0.3
    elif Q0 <= Q and Q < Q30:
        return 10.1
    elif Q30 <= Q and Q < Q70:
        return 14.4
    else:
        return 20.8

def load_model(modelpath):
    return tf.keras.models.load_model(modelpath)
