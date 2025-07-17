import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde 
import numpy as np
import matplotlib.cm as cm
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, roc_curve,auc,precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score
import math


def plot_boxplot_and_bar(df_nan,num_cols,cat_cols,target = "Personality"):
    categories = sorted(df_nan[target].unique())
    
    palette = sns.color_palette("Set2", len(categories))
    palette_dict = dict(zip(categories, palette))
    
    fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(10,3*math.ceil(len(num_cols)/3)))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(x=target, y=col, data=df_nan, ax=axes[i],order=categories, palette=palette_dict)
        axes[i].set_title(f"{col} by {target}")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(math.ceil(len(cat_cols)/3), 3, figsize=(10, 3 * math.ceil(len(cat_cols)/3)))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        x_order = sorted(df_nan[col].dropna().unique())  # 指定 x 轴顺序
        sns.countplot(
            x=col, hue=target, data=df_nan, ax=axes[i],
            order=x_order,                      # 固定 x 轴顺序
            hue_order=categories,               # 固定 hue 顺序
            palette=palette_dict                # 固定颜色
        )
        axes[i].set_title(f"{col} by {target}")
        axes[i].tick_params(axis='x', rotation=30) 

    plt.tight_layout()
    plt.show()
    
def plot_pdf(data, bins='auto', title='Probability Density Function', xlabel='Value', ylabel='Density'):
    """
    Plots the Probability Density Function (PDF) of the given data.
    
    Parameters:
    - data: array-like, the input data for which to plot the PDF.
    - bins: int or sequence or str (optional), the method for calculating bins for the histogram (default 'auto').
    - title: str (optional), the title of the plot.
    - xlabel: str (optional), the label for the x-axis.
    - ylabel: str (optional), the label for the y-axis.
    """
    plt.figure(figsize=(7, 4))
    sns.histplot(data, bins=bins, kde=True, stat='density')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    

def plot_curve(losses, title='Loss Curve', xlabel='Epoch', ylabel='Loss'):
    """
    loss curve plot, plot a single list
    - losses: list of losses
    """
    plt.figure(figsize=(8, 5))  
    plt.plot(losses, label='Training Loss', color='blue') 
    plt.title(title)  
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.legend()  
    plt.grid(True)  
    plt.show()  


def plot_loss_curves(train_losses, val_losses, title='Training and Validation Loss', xlabel='Epoch', ylabel='Loss'):
    """
    plot train and val loss

    """
    plt.figure(figsize=(10, 6))  
    plt.plot(train_losses, label='Training Loss', color='blue')  
    plt.plot(val_losses, label='Validation Loss', color='red')  
    plt.title(title)  
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.legend()  
    plt.grid(True)  
    plt.show()  



def plot_covariance_matrix(X, figsize=(8, 6), title='Covariance Matrix Heatmap'):
    """
    绘制给定数据集X的协方差矩阵热图。
    
    参数:
    - X: 数据集，其中行代表样本，列代表特征。
    - figsize: 热图的大小，以英寸为单位。
    - title: 热图的标题。
    """
    # 计算协方差矩阵
    X = np.array(X)
    covariance_matrix = np.cov(X.T)

    # 绘制热图
    plt.figure(figsize=figsize)
    sns.heatmap(covariance_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.show()
    
def plot_covariance(covariance_matrix, figsize=(6, 4), title='Covariance Matrix Heatmap'):
    """
    绘制给定协方差矩阵的热图。

    """
    plt.figure(figsize=figsize)
    sns.heatmap(covariance_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.show()

def plot_pdf_multiple_features(data, features_per_row=4):
    """
    Plot the probability density function for each feature in the data array.
    
    Parameters:
        data (2D array-like): The dataset where each row is a sample and each column a feature.
        features_per_row (int): The number of plots per row; defaults to 4.
    
    Returns:
        None: Plots the PDFs of the features.
    """
    num_features = data.shape[1]
    num_rows = (num_features + features_per_row - 1) // features_per_row  # Calculate required number of rows
    fig, axes = plt.subplots(num_rows, features_per_row, figsize=(features_per_row*4, num_rows*3))
    
    for i in range(num_features):
        ax = axes[i // features_per_row, i % features_per_row]
        feature_data = data[:, i]
        kde = gaussian_kde(feature_data)
        x = np.linspace(feature_data.min(), feature_data.max(), 500)
        ax.plot(x, kde(x), label=f'Feature {i+1}')
        ax.set_title(f'Feature {i+1}')
        ax.legend()

    # Hide any unused subplots
    for j in range(i+1, num_rows * features_per_row):
        axes[j // features_per_row, j % features_per_row].set_visible(False)

    plt.tight_layout()
    plt.show()
    

def plot_pdf_combined_classes(data, labels, class_names, features_per_row=4, linespace = 50, bw_method=0.2):
    """
    Plot the probability density function for each feature in the data array, combining all classes on a single plot per feature.
    
    Parameters:
        data (2D array-like): The dataset where each row is a sample and each column a feature.
        labels (array-like): The class labels for each row in data.
        class_names (list of str): List of names for the classes corresponding to unique labels.
        features_per_row (int): The number of plots per row; defaults to 4.
        bw_method (float, str, or callable, optional): The method used to calculate the estimator bandwidth. 
            This can directly affect the smoothness of the density estimate. Default is None, which uses the 
            Scott’s rule.
    
    Returns:
        None: Plots the PDFs of the features with all class distributions combined per plot.
    """
    unique_classes = np.unique(labels)
    num_features = data.shape[1]
    num_rows = (num_features + features_per_row - 1) // features_per_row  # Calculate required number of rows
    fig, axes = plt.subplots(num_rows, features_per_row, figsize=(features_per_row * 5, num_rows * 4))
    
    colors = cm.jet(np.linspace(0, 1, len(unique_classes)))  # Generate distinct colors for each class

    for i in range(num_features):
        ax = axes[i // features_per_row, i % features_per_row]
        for idx, cls in enumerate(unique_classes):
            class_data = data[labels == cls, i]
            if class_data.size > 0:  # Only plot if data is not empty
                kde = gaussian_kde(class_data, bw_method=bw_method)
                x = np.linspace(class_data.min(), class_data.max(), linespace)
                ax.plot(x, kde(x), label=f'{class_names[cls]}', color=colors[idx])
                ax.set_title(f'Feature {i+1}')
        ax.legend()

    # Hide any unused subplots
    for j in range(num_features, num_rows * features_per_row):
        axes[j // features_per_row, j % features_per_row].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_qq(data, feature_names, rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()  # 将axes数组扁平化，方便使用单个循环进行索引操作
    
    for i, col in enumerate(feature_names):
        stats.probplot(data[col], dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ plot for {col}')
    
    # 如果特征少于subplot数量，清除多余的subplot
    for ax in axes[i+1:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    

def plot_qq_array(data, num_features, rows, cols):
    """
    Plots QQ plots for multiple features given a 2D NumPy array.
    
    Parameters:
    - data: 2D NumPy array, where each column is a feature to plot.
    - num_features: integer, number of features to plot.
    - rows: integer, number of rows in the subplot grid.
    - cols: integer, number of columns in the subplot grid.
    
    Each feature in the data array will be plotted in a QQ plot to assess normality against the theoretical normal distribution.
    """
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()  # Flatten the axes array to facilitate indexing in a single loop
    
    for i in range(num_features):
        stats.probplot(data[:, i], dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ plot for feature {i+1}')
    
    # Clear any extra subplots if there are fewer features than subplots
    for ax in axes[num_features:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_test, y_prob):
    """
    Plots the Precision-Recall curve and calculates the PR AUC.

    Parameters:
    y_test (array-like): True binary labels.
    y_prob (array-like): Estimated probabilities or decision function.

    Returns:
    None
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()


def plot_roc_auc(y_test, y_pred_prob):
    """
    绘制ROC曲线并计算AUC值。
    
    参数:
    y_test (array-like): 真实标签。
    y_pred_prob (array-like): 预测的正类概率。
    
    返回:
    None
    """
    # 确保输入为NumPy数组
    y_test = np.asarray(y_test)
    y_pred_prob = np.asarray(y_pred_prob)

    # 计算AUC值
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    # 打印AUC值
    print(f"ROC AUC: {roc_auc}")


def plot_multiclass_roc_auc(y_test, y_pred_proba, class_names):
    """
    Plots the ROC AUC curve for a multi-class classification problem.

    Parameters:
    y_test (array-like): True labels.
    y_pred_proba (array-like): Predicted probabilities.
    class_names (list): List of class names corresponding to each label.

    Returns:
    None
    """
    n_classes = len(np.unique(y_test))
    
    # Binarize the true labels
    y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))
    
    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute micro-average ROC curve and ROC AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=2, label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.4f})')
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=2, label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.4f})')
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_multiclass_pr_auc(y_test, y_pred_proba, class_names):
    """
    Plots the Precision-Recall AUC curve for a multi-class classification problem.

    Parameters:
    y_test (array-like): True labels.
    y_pred_proba (array-like): Predicted probabilities.
    class_names (list): List of class names corresponding to each label.

    Returns:
    None
    """
    n_classes = len(np.unique(y_test))
    
    # Binarize the true labels
    y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))
    
    # Compute Precision-Recall curve and PR AUC for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        pr_auc[i] = average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i])

    # Compute macro-average PR AUC
    precision["macro"], recall["macro"], _ = precision_recall_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
    pr_auc["macro"] = average_precision_score(y_test_binarized, y_pred_proba, average="macro")

    # Compute micro-average PR AUC
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
    pr_auc["micro"] = average_precision_score(y_test_binarized, y_pred_proba, average="micro")

    # Plot PR curves
    plt.figure(figsize=(8, 6), dpi=100)
    plt.step(recall['macro'], precision['macro'], where='post', color='b', alpha=0.2, label=f'Macro-average PR curve (AUC = {pr_auc["macro"]:.4f})')
    plt.step(recall['micro'], precision['micro'], where='post', color='r', alpha=0.2, label=f'Micro-average PR curve (AUC = {pr_auc["micro"]:.4f})')
    for i in range(n_classes):
        plt.step(recall[i], precision[i], where='post', label=f'PR curve of class {class_names[i]} (AUC = {pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

