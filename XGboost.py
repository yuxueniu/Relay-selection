import numpy as np
from matplotlib import pyplot as plt

import nn_relay_util as util
import nn_relay_verify as v
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
#import pandas as pd
from timeit import default_timer as timer
import time

def XGboost_train(data_size=None, train_data=None, train_label=None, val_data=None, val_label=None):
    start = time.perf_counter()

    if data_size is None:
        data_size = 10000000
    if train_data is None and train_label is None and val_data is None and val_label is None:
        # get training and validation data
        train_data, train_label, train_ori = util.get_data_max_values(data_size)
        val_data, val_label, val_ori = util.get_data_max_values(data_size / 50)


    model = MultiOutputRegressor(xgb.XGBRegressor(max_depth=15, learning_rate=0.1, n_estimators=100))
    model.fit(train_data, train_label)
    pre = model.predict(val_data)

    pre = np.array(pre)
    val_label = np.array(val_label)
    fig = plt.figure()
    plt.scatter(pre[:, 0], pre[:, 1], c='#FFD700', alpha=0.4, marker='o', s=18, cmap='coolwarm', label='XGboost-pre')
    plt.scatter(val_label[:, 0], val_label[:, 1], c='#ff7f0e', alpha=0.4, marker='^', s=24, cmap='coolwarm', label='True label')
    plt.legend(loc="lower left",fontsize=10)
    plt.xticks(fontsize=10) #x轴刻度字体大小
    plt.yticks(fontsize=10) #y轴刻度字体大小  

    plt.savefig('XGboost.pdf')
    #plt.show()

    total_time = time.perf_counter() - start
    print("total time_XGB:", time.perf_counter() - start)
    return model,total_time

if __name__ == "__main__":

    model, total_time = XGboost_train()
    # validation
    #gamma_th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #for gamma in gamma_th:
    #    print("gamma", gamma)
    #    MSE, acc, out_prob_bf, out_prob_predict, op_random, op_bulk, op_persub = v.verify_with_outage_prob(model=model, gamma_th=gamma)
    #    print("out_prob_predict", out_prob_predict)
    #    print("acc", acc)
    #    #print("total time:", time.perf_counter() - start)
    # validation
    MSE, acc, out_prob_bf, out_prob_predict, op_random, op_bulk, op_persub, op_maxsnr, op_nearest, verify_latency = v.verify_with_outage_prob(size=10000000,model=model)
    print("out_prob_predict_XGB", out_prob_predict)
    print("acc_XGB", acc)
    print("MSE_XGB", MSE)
