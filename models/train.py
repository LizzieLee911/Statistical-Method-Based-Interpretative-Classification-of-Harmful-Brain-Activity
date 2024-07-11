import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde 
import numpy as np
import matplotlib.cm as cm
import scipy.stats as stats
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

def logistic_regression_analysis(X_train, y_train, X_test, y_test, 
                                 penalty='l1', C=1000, solver='saga', max_iter=1000, 
                                 class_weight='balanced', tol=1e-4, verbose=-1, n_jobs=-1,
                                 plot_roc=False):
    """
    Train and evaluate a logistic regression model.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - y_test: Test labels
    - penalty: Regularization type (default 'l1')
    - C: Inverse regularization strength (default 1000)
    - solver: Optimization algorithm (default 'saga')
    - max_iter: Maximum number of iterations (default 1000)
    - class_weight: Class weights (default 'balanced')
    - tol: Tolerance for stopping criteria (default 1e-4)
    - verbose: Verbosity level (default 1)
    - n_jobs: Number of CPU cores to use (default -1)
    - plot_roc: Whether to plot ROC curve (default True)
    - y_prob:IMPORTANT!!!This only returns the probs of poitive class
    """
    
    # Train logistic regression model
    log_reg = LogisticRegression(
        penalty=penalty,            
        C=C,                  
        solver=solver,         
        max_iter=max_iter,            
        class_weight=class_weight, 
        tol=tol,                
        verbose=verbose,               
        n_jobs=n_jobs                
    )
    log_reg.fit(X_train, y_train)
    
    # Predict
    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:,1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    

    # Print results
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"AUC: {auc}")

    # Plot ROC curve
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()
        
    return log_reg,y_pred,y_prob


def train_catboost_classifier(X_train, y_train, X_val, y_val, X_test, y_test, **kwargs):
    """
    Train a CatBoost classifier with specified parameters.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.
    X_val (array-like): Validation features.
    y_val (array-like): Validation labels.
    X_test (array-like): Testing features.
    y_test (array-like): Testing labels.
    kwargs (dict): Additional parameters for CatBoostClassifier.

    Returns:
    y_pred (array-like): Predicted labels for the test set.
    y_pred_proba (array-like): Predicted probabilities for the test set.
    """
    
    # Create Pool objects for training, validation, and testing data
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    test_pool = Pool(X_test, y_test)

    # Initialize the CatBoost classifier with given parameters
    model = CatBoostClassifier(**kwargs)

    # Train the model
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50  # Stop training if no improvement for 50 rounds
    )

    # Predict probabilities and labels for the test set
    y_pred_proba = model.predict_proba(test_pool)
    y_pred = model.predict(test_pool)

    return model, y_pred, y_pred_proba


