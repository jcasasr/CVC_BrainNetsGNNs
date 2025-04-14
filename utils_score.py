import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def compute_auc(target, preds, logger=None):
    # Compute metrics
    auc_roc = roc_auc_score(target, preds)
    auc_pr = average_precision_score(target, preds)
    prop = np.sum(target) / len(target)
        
    best_acc = 0
    best_th = 0
    for th in preds:
        acc = accuracy_score(target, (preds >= th).astype(int))
        if acc >= best_acc:
            best_acc = acc
            best_th = th
    # Compute confusion matrix
    conf_matrix = confusion_matrix(target, (preds >= best_th).astype(int))
            
    logger.info("")
    logger.info("AUC ROC  : {:.4f}".format(auc_roc))
    logger.info("AUC PR   : {:.4f}".format(auc_pr))
    logger.info("ACC      : {:.4f}".format(best_acc))
    logger.info("% of pwMS: {:.4f}".format(prop))
    logger.info("Confusion matrix:")
    logger.info(conf_matrix)


def plot_auc_roc(target, preds, name_out:str):
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.rcParams['font.size'] = 14

    prop = np.sum(target) / len(target)

    # compute AUC-ROC and ROC curve
    auc_roc = roc_auc_score(target, preds)
    fpr, tpr, _ = roc_curve(target, preds)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (AUC-ROC: {:.4f})".format(auc_roc))
    plt.savefig(os.path.join("figs", name_out +"_ROC.png"), dpi=300, bbox_inches='tight', pad_inches=0)

    # Compute AUC-PR
    auc_pr = average_precision_score(target, preds)
    prec, recall, _ = precision_recall_curve(target, preds)

    plt.figure()
    lw = 2
    plt.plot(recall, prec, color="darkorange", lw=lw)
    plt.plot([0, 1], [prop, prop], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve (AUC-PR: {:.4f})".format(auc_pr))
    plt.savefig(os.path.join("figs", name_out +"_P-R.png"), dpi=300, bbox_inches='tight', pad_inches=0)