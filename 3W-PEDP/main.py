import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.datasets import make_blobs
from collections import Counter
from scipy.spatial import distance
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
import csv
import math
from sklearn.preprocessing import MinMaxScaler
from collections import deque
np.set_printoptions(suppress=True)
import time

# 聚类结果展示
def draw_cluster_results(datas, label_pred, centroid):
    plt.figure()
    plt.scatter(datas[:, 0], datas[:, 1], c=label_pred, marker='o')
    for i in range(len(centroid)):
        plt.scatter(datas[centroid[i], 0], datas[centroid[i], 1], c=label_pred[centroid[i]], marker='+', s=300)
    plt.show()


# 读取数据集
def read_data(data_name):
    with open('dataset/' + data_name + '.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    dataset = np.array(dataset).astype(float)
    label1 = dataset[:, 0]
    label1 = label1.astype(int)
    data = dataset[:, 1:]
    return label1, data


# 评价效果
def evaluation_effect(label_true, label_pred):
    clusters = np.unique(label_pred)
    labels_true = np.reshape(label_true, (-1, 1))
    labels_pred = np.reshape(label_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    acc = np.sum(count) / labels_true.shape[0]
    print(f"ACC = {acc:.4f}")
    print(f"ARI = {adjusted_rand_score(label_true, label_pred):.4f}")
    # print(f"AMI = {adjusted_mutual_info_score(label_true, label_pred):.4f}")
    print(f"FMI = {fowlkes_mallows_score(label_true, label_pred):.4f}")



# 欧式距离
def getDistanceMatrix(datas):
    dists = cdist(datas, datas, metric='euclidean')
    return dists

def rgK2(dists):
    length = np.shape(dists)[0]
    num = int(math.sqrt(np.shape(dists)[0]))
    sortDist = np.sort(dists)
    indexDist = np.argsort(dists)
    cov = np.zeros([length, num])
    sp = np.zeros([length, num])
    numK = np.zeros([length, 1])
    for i in range(length):
        for j in range(num):
            mean = np.sum(sortDist[i, :j+1]) / (j + 1)
            cov[i, j] = np.sum(np.exp(-np.abs(sortDist[i, :j + 1] - mean)))/num
            sp[i, j] = 1 - (j+1)/num
    q = cov * sp
    # +1因为索引从0开始
    for i in range(length):
        numK[i] = np.argmax(q[i, :]) + 1
    numK = list(map(int, numK))
    kVal2 = Counter(numK).most_common(1)[0][0]
    return kVal2


# 局部密度
def get_rho(dists, kVal):
    dist = np.sort(dists)
    sortDist = dist[:, :kVal]
    rho = np.zeros(np.shape(dists)[0])
    for i in range(np.shape(dists)[0]):
        rho[i] = np.sum(np.log(2 / (sortDist[i, :] + 1)))
        # rho[i] = np.sum(np.log(2/(sortDist[i, :]+1)+1))
        # rho[i] = np.sum(np.exp(-sortDist[i, :]))
        # rho[i] = np.sum(np.log(2 / np.exp(-sortDist[i, :])))
        if rho[i] < 0:
            rho[i] = 0
    return rho


# 聚类中心选取图
def draw_decision(rho, deltas):
    plt.cla()
    for i in range(np.shape(rho)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    plt.show()


# 中心偏移距离， 最近邻居
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])

        # 保存最近邻点的编号

        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


# 聚类中心筛选
def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def interNN(indexDist, a, nei, kVal):
    b = 0
    que = np.zeros([2])
    que[0] = a
    que[1] = nei
    for j in range(kVal):
        # if len(np.intersect1d(que, indexDist[indexDist[nei, j], :])) >= 2:
        if len(np.intersect1d(indexDist[a, :], indexDist[indexDist[nei, j], :])) >= kVal/2:
            b += 1
    return b


# 分配核心点
def assign_core(dists, kVal, centers):
    indexDist = np.argsort(dists)
    indexDist = indexDist[:, :kVal]
    sortDist = np.sort(dists)
    label = np.zeros([np.shape(dists)[0], 2])
    queue = deque()
    # 把聚类中心点加入至队列中
    for i in range(len(centers)):
        label[centers[i], 0] = i
        label[centers[i], 1] = -1
        queue.append(centers[i])
    while queue:
        a = queue[0]
        for j in range(kVal):
            b = interNN(indexDist, a, indexDist[a, j], kVal)
            if label[indexDist[a, j], 1] == 0 and b >= kVal/2:
                label[indexDist[a, j], 0] = label[a, 0]
                label[indexDist[a, j], 1] = -1
                queue.append(indexDist[a, j])
        queue.popleft()
    return label



# 未分配点
def no_group(label):
    unAssign = deque()
    for i in range(np.shape(label)[0]):
        if label[i, 1] == 0:
            unAssign.append(i)
            # label[i, 0] = 4
    return unAssign


#  分配边缘点
def assign_no_group(unAssign, label, dists, centerNum, kVal):
    indexDist = np.argsort(dists)
    sortDist = np.sort(dists)
    sortDist = np.exp(-sortDist)
    if len(indexDist[0]) > 1000:
        num = 100
    else:
        num = 50
    while no_group(label):
        if kVal < num:
            for i in range(len(unAssign)):
                # 计算质量函数
                mass = np.zeros([centerNum + 1, kVal])
                for j in range(centerNum + 1):
                    for t in range(kVal):
                        if j != centerNum:
                            nei = indexDist[unAssign[i], t]
                            index = np.where(label[indexDist[nei, :kVal], 1] == -1)[0]
                            length = np.where(label[indexDist[nei, index], 0] == j)[0].shape[0]
                            dis = sortDist[unAssign[i], t]
                            mass[j, t] = dis * (length / kVal)
                        else:
                            mass[centerNum, t] = 1 - np.sum(mass[:centerNum, t])
                # 质量融合
                mass_Fusion = np.zeros([centerNum + 1])
                for j in range(centerNum + 1):
                    if j != centerNum:
                        mass_Fusion[j] = np.prod(mass[j, :] + mass[centerNum, :]) - np.prod(mass[centerNum, :])
                    else:
                        mass_Fusion[centerNum] = np.prod(mass[centerNum, :])
                mass_Fusion[:] = mass_Fusion[:] / np.sum(mass_Fusion[:])
                possible = mass_Fusion[:-1]
                if np.max(possible[:]) > 0.5:
                    label[unAssign[i], 0] = np.argsort(-possible[:])[0]
                    label[unAssign[i], 1] = -1
                elif kVal < num:
                    kVal += 1
        else:
            break
    while no_group(label):
        unAssign2 = no_group(label)
        for x in range(len(unAssign2)):
            for j in range(np.shape(dists)[0]):
                if label[indexDist[unAssign2[x], j], 1] == -1:
                    label[unAssign2[x], 0] = label[indexDist[unAssign2[x], j], 0]
                    label[unAssign2[x], 1] = -1
    return label


if __name__ == "__main__":
    start_time = time.time()
    # 数据集名称
    dataset_name = 'wine'
    # 聚类个数
    centerNum = 3
    # 标签, 数据
    label_true, datas = read_data(dataset_name)
    # 数据最大最小归一化
    datas = MinMaxScaler().fit_transform(datas)
    # 距离矩阵
    dists = getDistanceMatrix(datas)
    # 合理邻居个数
    kVal = rgK2(dists)
    # 局部密度
    rho = get_rho(dists, kVal)
    # 计算密度距离
    deltas, nearest_neighbor = get_deltas(dists, rho)
    centers = find_centers_K(rho, deltas, centerNum)
    draw_decision(rho, deltas)
    label = assign_core(dists, kVal, centers)
    unAssign = no_group(label)
    label = assign_no_group(unAssign, label, dists, centerNum, kVal)
    draw_cluster_results(datas, label[:, 0], centers)
    evaluation_effect(label_true, label[:, 0])
    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间：", run_time)
    # 邻居个数从3-40
    for i in range(3, 40):
        print("K:", i, "----------------------")
        kVal = i
        rho = get_rho(dists, kVal)
        deltas, nearest_neighbor = get_deltas(dists, rho)
        centers = find_centers_K(rho, deltas, centerNum)
        label = assign_core(dists, kVal, centers)
        unAssign = no_group(label)
        label = assign_no_group(unAssign, label, dists, centerNum, kVal)
        evaluation_effect(label_true, label[:, 0])
        print("--------------------------------")