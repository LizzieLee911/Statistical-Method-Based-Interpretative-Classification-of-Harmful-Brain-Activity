import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_ind

def standarlize_spe_min_max(processed_data):
    mn = processed_data.min()
    mx = processed_data.max()
    ep = 1e-5
    return (processed_data - mn) / (mx - mn + ep)

def standardize_spe_z_score_per_feature(processed_data, axis = 1):
    mean = processed_data.mean(axis=axis, keepdims=True)
    std = processed_data.std(axis=axis, keepdims=True)
    return (processed_data - mean) / (std + 1e-5)

def standardize_spe_z_score(processed_data):
    mean = processed_data.mean()
    std = processed_data.std()
    return (processed_data - mean) / (std + 1e-5)

def SAMPLERrep_ver2(X, axis=0, pers=[5,15,25,35,45,55,65,75,85,95]):
    '''Return SAMPLER from a 2D array, ignoring NaN values, be careful of the output shape it'll freaking Transpose'''
    # 如果X全是NaN值，则用0填充
    if np.count_nonzero(~np.isnan(X)) == 0:
        X = np.nan_to_num(X, nan=0.0)
    SAMPLER_rep = np.nanpercentile(X, pers, axis)
    return SAMPLER_rep


def standardize_sample(sample,axis=0):
    """standarlize per colomn"""
    mean = np.mean(sample, axis)
    std = np.std(sample, axis)
    standardized_sample = (sample - mean) / (std + 1e-8)  # 防止除以零
    return standardized_sample

import numpy as np

def kl_divergence(p, q):
    """
    计算两个一维数组之间的KL散度。
    
    参数:
    - p: 第一个概率分布，一维numpy数组。
    - q: 第二个概率分布，一维numpy数组。
    
    返回:
    - KL散度值。
    """
    # 确保p和q都是有效的概率分布（和为1，非负）
    p = np.array(p)
    q = np.array(q)
    
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    
    # 防止分母是0
    q = np.where(q == 0, 1e-12, q)
    kl_div = np.sum(p * np.log(p / q))
    
    return kl_div

def mahalanobis_distance(x, y, inv_covariance_matrix):
    """
    使用已计算的协方差矩阵的逆（或伪逆）计算两个向量之间的马氏距离。
    
    参数:
    - x, y: 两个N维向量。
    - inv_covariance_matrix: 数据集协方差矩阵的逆（或伪逆）。
    
    返回:
    - x和y之间的马氏距离。
    """
    diff_vector = x - y
    distance = np.sqrt(np.dot(np.dot(diff_vector.T, inv_covariance_matrix), diff_vector))
    return distance

def softmax(x):
    """
    计算softmax函数。
    
    参数:
    - x: NumPy数组，可以是向量或矩阵。
    
    返回:
    - softmax后的NumPy数组。
    """
    # 防止指数计算溢出，减去每行的最大值
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

from sklearn.preprocessing import LabelEncoder
import numpy as np



def print_encoder_contents(encoder):
    """
    Print the classes and their corresponding integer labels from a LabelEncoder.
    
    Parameters:
        encoder (LabelEncoder): The LabelEncoder instance to inspect.
    
    Returns:
        None: Prints the contents of the encoder.
    """
    # Ensure that the encoder has been fitted
    if hasattr(encoder, 'classes_'):
        # Iterate over classes and their indexes (which are the encoded labels)
        for index, label in enumerate(encoder.classes_):
            print(f"Original Label: {label} -> Encoded Label: {index}")
    else:
        print("The encoder has not been fitted yet.")


def calculate_kl_divergence(logits, targets):
    """
    计算并打印由numpy数组或列表输入的原始概率logits和真实概率targets的KL散度。
    
    参数:
    logits (array-like): 原始概率形式的预测值。
    targets (array-like): 真实的概率分布。
    """
    # 将输入的 numpy 数组或列表转换为 PyTorch 张量
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    # 将原始概率 logits 转换为对数概率
    log_probs = F.log_softmax(logits_tensor, dim=1)

    # 初始化 KL 散度损失函数
    criterion = nn.KLDivLoss(reduction='batchmean')

    # 计算 KL 散度
    kl_divergence = criterion(log_probs, targets_tensor)

    # 打印 KL 散度结果
    print("KL Divergence:", kl_divergence.item())



def perform_t_test(X_combined, y, positive_class = 0, alpha=0.05):
    """
    Perform t-test for each feature in X_combined to determine significant differences between positive and negative classes.

    Parameters:
    X_combined (np.ndarray): Feature matrix.
    y (np.ndarray): Target vector.
    alpha (float): Significance level.

    Returns:
    list: Features to consider removing based on p-value.
    """
    # Identify positive and negative classes
    positive_class = positive_class
    positive_indices = (y == positive_class)
    negative_indices = (y != positive_class)

    # Initialize list to store features to remove
    features_to_remove = []

    # Perform t-test for each feature
    for feature_idx in range(X_combined.shape[1]):
        positive_values = X_combined[positive_indices, feature_idx]
        negative_values = X_combined[negative_indices, feature_idx]

        t_stat, p_value = ttest_ind(positive_values, negative_values, equal_var=False, nan_policy='omit')

        if p_value > alpha:
            features_to_remove.append(feature_idx)

    print("Features to consider removing (p > alpha):")
    print(features_to_remove)
    print(f"Number of features to consider removing: {len(features_to_remove)}")

    return features_to_remove




