import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import statsmodels.api as sm
import pandas as pd
import pingouin as pg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ConfusionMatrix(cm, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix
    # cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    font_S = fm.FontProperties(family='DejaVu Sans', size=12, stretch=0)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           # title=title,
           ylabel='True Sleep Stage',
           xlabel='Predicted Sleep Stage',
           )
    ax.set_xlabel(xlabel='Predicted Sleep Stage', fontsize=12, fontproperties=font_S)
    ax.set_ylabel(ylabel='True Sleep Stage', fontsize=12, fontproperties=font_S)
    ax.set_xticklabels(labels=classes, fontproperties=font_S)
    ax.set_yticklabels(labels=classes, fontproperties=font_S)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    plt.rc('font', family='DejaVu Sans')
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] * 100, '.2f') + '%\n' + format(cm_n[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath + '/' + title + ".png", dpi=600, bbox_inches='tight')
    # plt.show()
    return ax


def PrintScore(true, pred, num_classes, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    if num_classes == 3:
        classes = ['Awake', 'NREM', 'REM']
        print('Acc_\tF1_S\tKappa\tF1_W\tF1_NR\tF1_R', file=saveFile)
        print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
            metrics.accuracy_score(true, pred),
            metrics.f1_score(true, pred, average=average),
            metrics.cohen_kappa_score(true, pred),
            F1[0], F1[1], F1[2]),
            file=saveFile)
    elif num_classes == 2:
        classes = ['Awake', 'Sleep']
        print('Acc_\tF1_S\tKappa\tF1_W\tF1_S', file=saveFile)
        print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
            metrics.accuracy_score(true, pred),
            metrics.f1_score(true, pred, average=average),
            metrics.cohen_kappa_score(true, pred),
            F1[0], F1[1]),
            file=saveFile)
    elif num_classes == 4:
        classes = ['Awake', 'Light', 'Deep', 'REM']
        print('Acc_\tF1_S\tKappa\tF1_W\tF1_L\tF1_D\tF1_R', file=saveFile)
        print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
            metrics.accuracy_score(true, pred),
            metrics.f1_score(true, pred, average=average),
            metrics.cohen_kappa_score(true, pred),
            F1[0], F1[1], F1[2], F1[3]),
            file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)

    print(metrics.classification_report(true, pred,
                                        target_names=classes,
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    if savePath != None:
        saveFile.close()
    return


def PrintEventScore(true, pred, num_classes, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "EventResult.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    if num_classes == 6:
        classes = ['Normal', 'CSA', 'Hypopnea', 'MSA', 'OSA', 'Unsure']
        print('Acc_\tF1_S\tKappa\tF1_N\tF1_C\tF1_H\tF1_M\tF1_O\tF1_U', file=saveFile)
        print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
            metrics.accuracy_score(true, pred),
            metrics.f1_score(true, pred, average=average),
            metrics.cohen_kappa_score(true, pred),
            F1[0], F1[1], F1[2], F1[3], F1[4], F1[5]),
            file=saveFile)
    elif num_classes == 5:
        classes = ['Normal', 'CSA', 'Hypopnea', 'MSA', 'OSA']
        print('Acc_\tF1_S\tKappa\tF1_N\tF1_C\tF1_H\tF1_M\tF1_O', file=saveFile)
        print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
            metrics.accuracy_score(true, pred),
            metrics.f1_score(true, pred, average=average),
            metrics.cohen_kappa_score(true, pred),
            F1[0], F1[1], F1[2], F1[3], F1[4]),
            file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)

    print(metrics.classification_report(true, pred,
                                        target_names=classes,
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    if savePath != None:
        saveFile.close()
    return


def GetClassWeight(train_rule, n_class, cls_num_train):
    if train_rule == 'None':
        per_cls_weights = np.array([1.0, 1.0, 1.0, 1.0])
    elif train_rule == 'ClassBalance':
        beta = 0.999999
        effective_num = 1.0 - np.power(beta, cls_num_train)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_train)
        # per_cls_weights = dict(enumerate(per_cls_weights))
    elif train_rule == 'ReWeight':
        per_cls_weights = 1 / np.array(cls_num_train)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * n_class
        # per_cls_weights = dict(enumerate(per_cls_weights))
    elif train_rule == 'SqrtReWeight':
        per_cls_weights = 1 / np.sqrt(np.array(cls_num_train))
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * n_class
        # per_cls_weights = dict(enumerate(per_cls_weights))
    else:
        warnings.warn('Train rule is not listed')
        return np.array([1.0, 1.0, 1.0, 1.0])
    return per_cls_weights


def AHIConfusionMatrix(y_true, y_pred, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion_matrix'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    font_S = fm.FontProperties(family='DejaVu Sans', size=12, stretch=0)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           # ylabel='True Sleep Stage',
           # xlabel='Predicted Sleep Stage',
           )
    ax.set_xlabel(xlabel='Predicted AHI degree', fontsize=12, fontproperties=font_S)
    ax.set_ylabel(ylabel='True AHI degree', fontsize=12, fontproperties=font_S)
    ax.set_xticklabels(labels=classes, fontproperties=font_S)
    ax.set_yticklabels(labels=classes, fontproperties=font_S)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    plt.rc('font', family='DejaVu Sans')
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] * 100, '.2f') + '%\n' + format(cm_n[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath + title + ".png", dpi=800)
    return ax


def PrintAHIScore(true, pred, savePath=None, average='macro'):
    # savePath=None -> console, else to Result.txt
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")

    classes = ['Normal', 'Mild', 'Moderate', 'Severe']
    print('Acc_\tF1_S\tKappa\tF1_Normal\tF1_Mild\tF1_Moderate\tF1_Severe', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
        metrics.accuracy_score(true, pred),
        metrics.f1_score(true, pred, average=average),
        metrics.cohen_kappa_score(true, pred),
        F1[0], F1[1], F1[2], F1[3]),
        file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)

    print(metrics.classification_report(true, pred,
                                        target_names=classes,
                                        digits=4), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    if savePath != None:
        saveFile.close()
    return


def PrintAHIResults(y_true, y_pred, savepath):
    saveFile = open(savepath + '/Result.txt', 'a+')
    y_pred = np.squeeze(y_pred)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print("AHI RMSE:", rmse)
    print("AHI RMSE:", rmse, file=saveFile)

    # 计算平均绝对误差（MAE）
    mae = np.mean(np.abs(y_true - y_pred))
    print("AHI MAE:", mae)
    print("AHI MAE:", mae, file=saveFile)

    # 计算相关系数（Pearson Correlation Coefficient）
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    print("AHI Correlation:", correlation)
    print("AHI Correlation:", correlation, file=saveFile)

    pred_ahi = y_pred
    true_ahi = y_true
    y_true_tmp = np.zeros((len(pred_ahi),), dtype=int)
    y_pred_tmp = np.zeros((len(pred_ahi),), dtype=int)
    y_pred_tmp[pred_ahi < 5] = 0
    y_pred_tmp[(pred_ahi >= 5) & (pred_ahi < 15)] = 1
    y_pred_tmp[(pred_ahi >= 15) & (pred_ahi < 30)] = 2
    y_pred_tmp[pred_ahi >= 30] = 3

    y_true_tmp[true_ahi < 5] = 0
    y_true_tmp[(true_ahi >= 5) & (true_ahi < 15)] = 1
    y_true_tmp[(true_ahi >= 15) & (true_ahi < 30)] = 2
    y_true_tmp[true_ahi >= 30] = 3

    PrintAHIScore(y_true_tmp, y_pred_tmp)
    Accuracy = round(metrics.accuracy_score(y_true_tmp, y_pred_tmp), 4)
    print("ahi class Acc:", Accuracy)
    print("ahi class Acc:", Accuracy, file=saveFile)

    # 计算 ICC（Intraclass Correlation Coefficient，类内相关系数）
    data = pd.DataFrame({'Predictions': y_pred, 'References': y_true})
    data_long = pd.melt(data.reset_index(), id_vars='index', value_vars=['Predictions', 'References'])
    data_long.columns = ['AHI', 'Type', 'Value']
    icc = pg.intraclass_corr(data=data_long, targets='AHI', raters='Type', ratings='Value').round(3)
    print(icc)
    print(icc, file=saveFile)
    
    r2 = metrics.r2_score(y_pred, y_true)
    print("R2:", r2)
    print("R2:", r2, file=saveFile)
    saveFile.close()

    AHIConfusionMatrix(y_true_tmp, y_pred_tmp, classes=['Normal', 'Mild', 'Moderate', 'Severe'], 
                       savePath=savepath)

    font_S = fm.FontProperties(family='DejaVu Sans', size=12, stretch=0)
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.rc('font', family='DejaVu Sans')
    sm.graphics.mean_diff_plot(y_pred, y_true, ax=ax[1], scatter_kwds={'c': y_true_tmp})
    scatter = ax[0].scatter(y_pred, y_true, c=y_true_tmp)
    handles, _ = scatter.legend_elements()
    ax[0].legend(handles, ['Normal', 'Mild', 'Moderate', 'Severe'])
    ax[0].plot([-10, 100], [-10, 100], 'r')
    ax[0].tick_params(labelsize=12)
    ax[0].set_xlabel(xlabel='Our AHI', fontsize=15, fontproperties=font_S)
    ax[0].set_ylabel(ylabel='PSG AHI', fontsize=15, fontproperties=font_S)
    ax[0].set_aspect('equal', adjustable='box')
    f.tight_layout()
    plt.show()
    return
