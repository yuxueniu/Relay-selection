import copy
import itertools
import multiprocessing
import random
import numpy as np
import time
import threading
import math
from multiprocessing.pool import ThreadPool
'''
这段代码定义了一些用于生成和操作与中继选择相关的数据的函数。
ExpConfig类定义了整个代码中使用的一些参数，例如中继数量、子载波数量和其他系统特定值。
Solution类用于表示给定一组中继和子载波的中继选择问题的解决方案。
get_max_values函数接受一个子载波值矩阵，并返回所有中继中每个子载波的最大值。
brute_force2函数接受相同的矩阵，并返回一个二进制值列表，指示应选择哪些中继以实现最大子载波值。
generate_G函数使用指数分布生成子载波值矩阵，然后应用subcarrier_min函数选择所有中继中每个子载波的最小值。
然后对结果矩阵进行归一化和排序，以生成每个子载波的唯一索引。
get_data函数使用generate_G和brute_force2函数生成子载波索引和相应的中继选择的数据集。
get_data_max_values函数生成类似的数据集，但使用get_max_values函数而不是brute_force2函数生成标签。
get_outage_prob函数计算给定一组子载波值和中继选择的中断发生概率。
总的来说，这段代码提供了一个框架，用于生成和分析与无线通信网络中继选择相关的数据集。
'''

class ExpConfig:
    M = 8
    K = 2 # 2
    mu1 = 1
    mu2 = 2
    L = 2
    Pt = 1
    N0 = 1
    gamma_th = 1


class Solution:
    def __init__(self, raw_table, combi, config=None):
        if config is None:
            config = ExpConfig()
        self.combi = combi
        self.raw_table = raw_table
        # first max
        a = raw_table[combi[0]]
        b = raw_table[combi[1]]
        c = copy.deepcopy(a)
        for i in range(config.K):
            if c[i] < b[i]:
                c[i] = b[i]
        self.first_max = c
        # print(first_max)
        # second min
        self.second_min = np.min(self.first_max)


def get_outage_prob(data, result, gamma_th=None):
    config = ExpConfig()
    if gamma_th is not None:
        config.gamma_th = gamma_th #added on 2023/01/01 to plot figure with the variance of gamma
    outage_count = 0
    for i in range(len(data)):
        tmp = []
        r = result[i]
        G = data[i]
        for m in range(config.M):
            if r[m] == 1:
                tmp.append(G[m])
        G_selected = np.max(np.array(tmp), axis=0)
        flag = config.Pt * np.min(G_selected) / config.N0

        if gamma_th is None:
            gamma_th = config.gamma_th

        if flag < gamma_th:
            outage_count += 1

    return outage_count / len(data)


def generate_G_idx():
    config = ExpConfig()
    G_idx = np.array(range(1, config.M * config.K + 1))
    random.shuffle(G_idx)
    G_min = G_idx.min()
    G_max = G_idx.max()
    # G_normalized = (G_idx - G_min) / (G_max - G_min)
    G_idx = np.reshape(G_idx, (8, 2)) # -------------------------------------------
    G_normalized = G_idx / G_max
    G_idx_origin = G_idx
    G_ori_normal = G_idx_origin / G_max
    return G_idx, G_normalized, G_idx_origin, G_ori_normal


def get_max_values(G):
    config = ExpConfig()
    tmp_combi = itertools.combinations(range(config.M), 2)
    solutions = []
    for t in tmp_combi:
        solutions.append(Solution(G, t))
    max_solution = max(solutions, key=lambda x: x.second_min)
    max_values = max_solution.first_max
    return max_values


def get_data_max_values(data_size, config=None):
    if config is None:
        config = ExpConfig()
    data = []
    orin_data = []
    label = []
    t0 = time.time()
    for i in range(round(data_size)):
        idx, idx_norm, idx_ori, idx_norm_ori = generate_G_idx()
        G_idx_full_reshape = np.reshape(idx_norm, [config.M * config.K])
        G_idx_ori_reshape = np.reshape(idx_norm_ori, [config.M, config.K])
        data.append(G_idx_full_reshape.tolist())
        orin_data.append((G_idx_ori_reshape.tolist()))
        tmp_label = get_max_values(idx_norm)
        # tmp_label = np.sort(tmp_label)
        label.append(tmp_label.tolist())
    t1 = time.time()
    gen_time = t1 - t0
    print("get_data: ", data_size, gen_time)
    return data, label, orin_data


def get_relay_by_selection(G, s):
    config = ExpConfig()
    S_d_relay = [0] * config.M
    for i in range(config.K):
        dis = 9999
        selected_id = -1
        for j in range(config.M):
            if abs(s[i] - G[j][i]) < dis:
                dis = abs(s[i] - G[j][i])
                selected_id = j
        S_d_relay[selected_id] += 1

    for i in range(len(S_d_relay)):
        if S_d_relay[i] > 1:
            S_d_relay[i] = 1

    return S_d_relay


def generate_G(config=None, target=None):
    if config == None:
        config = ExpConfig()
    G1 = np.random.exponential(scale=1, size=[config.M, config.K])
    G2 = np.random.exponential(scale=1, size=[config.M, config.K])
    G = subcarrier_min(G1, G2)
    G_min = G.min()
    G_max = G.max()
    G_normalized = 2 * (G - G_min) / (G_max - G_min) - 1
    G_reshape = np.sort(G.reshape(1, config.M * config.K))
    G_flip = np.flip(G_reshape)
    G_idx_full = copy.deepcopy(G)
    for i in range(config.M):
        for j in range(config.K):
            G_idx_full[i, j] = np.where(G_flip == G_idx_full[i, j])[1]
    # print(G)
    # print("*******************")
    # print(G_normalized)
    # print(G_idx_full)
    return G, G_normalized, G_idx_full / (config.M * config.K)


def subcarrier_min(a, b):
    c = copy.deepcopy(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if c[i, j] > b[i, j]:
                c[i, j] = b[i, j]
    return c


def brute_force2(G, config=None):
    if config is None:
        config = ExpConfig()
    max_values = get_max_values(G)
    S_d_relay = [0] * config.M
    for i in range(config.M):
        if max_values in G[i]:
            S_d_relay[i] += 1
    return S_d_relay


def get_idx(G):
    config = ExpConfig()
    G_idx_full = copy.deepcopy(G)
    G_reshape = np.sort(G.reshape(1, config.M * config.K))
    G_flip = np.flip(G_reshape)
    for i in range(config.M):
        for j in range(config.K):
            G_idx_full[i, j] = np.where(G_reshape == G_idx_full[i, j])[1]
    G_idx_full = G_idx_full + 1
    return G_idx_full


def get_data(config, data_size):
    data = []
    label = []
    t0 = time.time()
    for i in range(round(data_size)):
        G, G_normalized, G_idx_full = generate_G()
        G_idx_full_reshape = np.reshape(G_idx_full, [config.M * config.K])
        data.append(G_idx_full_reshape.tolist())
        bf2 = brute_force2(G_idx_full)
        S_d_relay = bf2
        label.append(S_d_relay)
    t1 = time.time()
    gen_time = t1 - t0
    print("get_data", data_size, gen_time)
    return data, label


if __name__ == "__main__":
    pass
