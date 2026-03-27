import numpy as np
import warnings
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn import metrics
from itertools import cycle
from prettytable import PrettyTable
import cv2 as cv
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():

    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'RSA-ADNet-LSTM', 'TFMOA-ADNet-LSTM', 'SCO-ADNet-LSTM', 'RTH-ADNet-LSTM',
                 'IF-RTH-ADNet-LSTM']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((5, 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report Dataset', 0 + 1,
          ' --------------------------------------------------')
    print(Table)

    length = np.arange(50)
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, label='RSA-ADNet-LSTM')
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, label='TFMOA-ADNet-LSTM')
    plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, label='SCO-ADNet-LSTM')
    plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, label='RTH-ADNet-LSTM')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label='IF-RTH-ADNet-LSTM')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence.png")
    plt.show()


def ROC_curve():
    lw = 2
    cls = ['RAN', 'Densenet', 'LSTM', 'ADNet-DenseNet', 'IF-RTH-ADNet-LSTM']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    colors = cycle(["#fe2f4a", "#0165fc", "#ffff14", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            # label=cls[i],
            label=f'{cls[i]} (AUC = {roc_auc:.2f})',
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_BatchSize():
    eval = np.load('Eval_ALL_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'MK', 'PLHR', 'Prevalence', 'TS']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Algorithm = ['TERMS', 'RSA-ADNet-LSTM', 'TFMOA-ADNet-LSTM', 'SCO-ADNet-LSTM', 'RTH-ADNet-LSTM', 'IF-RTH-ADNet-LSTM']
    Classifier = ['TERMS', 'RAN', 'Densenet', 'LSTM', 'ADNet-DenseNet', 'IF-RTH-ADNet-LSTM']
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- 48 Batch Size for Dataset', i + 1,
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- 48 Batch Size for Dataset', i + 1,
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
    for m in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[m, k, l, Graph_Term[j] + 4]

            length = np.arange(6)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='D', markerfacecolor='red',  # 98F5FF
                    markersize=6, label='RSA-ADNet-LSTM')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='s', markerfacecolor='green',  # 7FFF00
                    markersize=6, label='TFMOA-ADNet-LSTM')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='H', markerfacecolor='cyan',  # C1FFC1
                    markersize=8, label='SCO-ADNet-LSTM')
            ax.plot(length, Graph[:, 3], color='#00ffff', linewidth=3, marker='p', markerfacecolor='#fdff38',
                    markersize=8, label='RTH-ADNet-LSTM')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='w', markersize=12,
                    label='IF-RTH-ADNet-LSTM')
            plt.xticks(length, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch Size', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/BATCHSIZE_%s_lrean.png" % (Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(6)

            ax.bar(X + 0.00, Graph[:, 5], color='#0165fc', edgecolor='w', width=0.15, label="RAN")
            ax.bar(X + 0.15, Graph[:, 6], color='#ff474c', edgecolor='w', width=0.15, label="Densenet")
            ax.bar(X + 0.30, Graph[:, 7], color='#be03fd', edgecolor='w', width=0.15, label="LSTM")
            ax.bar(X + 0.45, Graph[:, 8], color='#21fc0d', edgecolor='w', width=0.15, label="ADNet-DenseNet")
            ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="IF-RTH-ADNet-LSTM")
            plt.xticks(X + 0.15, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch Size', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/BATCHSIZE_%s_bar.png" % (Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def plot_results_Seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Full = ['TERMS', 'Unet', 'Unet3+', 'TransUnet', 'Trans-Unet++', '3D-TDUNet++']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i]) * 100
                    stats[i, j, 1] = np.min(value_all[j][:, i]) * 100
                    stats[i, j, 2] = np.mean(value_all[j][:, i]) * 100
                    stats[i, j, 3] = np.median(value_all[j][:, i]) * 100
                    stats[i, j, 4] = np.std(value_all[j][:, i]) * 100

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax.bar(X + 0.00, stats[i, 0, :], color='#ff028d', edgecolor='k', width=0.10, label="Unet")  # r
            ax.bar(X + 0.10, stats[i, 1, :], color='#0cff0c', edgecolor='k', width=0.10, label="Unet3+")  # g
            ax.bar(X + 0.20, stats[i, 2, :], color='#0165fc', edgecolor='k', width=0.10, label="TransUnet")  # b
            ax.bar(X + 0.30, stats[i, 3, :], color='#fd411e', edgecolor='k', width=0.10, label="Trans-Unet++")  # m
            ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='k', width=0.10, label="3D-TDUNet++")  # k
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path = "./Results/Segmentation_%s_mtd.png" % (Terms[i - 4])
            plt.savefig(path)
            plt.show()


def Image_segment_comparision():
    Original = np.load('Images.npy', allow_pickle=True)
    segmented = np.load('Seg_img.npy', allow_pickle=True)
    Ground_truth = np.load('GroundTruth.npy', allow_pickle=True)
    Image = [760, 1362, 1364, 1865, 3075]
    for i in range(len(Image)):
        Orig = Original[Image[i]]
        Seg_1 = segmented[Image[i]]
        GT = Ground_truth[Image[i]]
        for j in range(1):
            # print(i, j)
            Orig_1 = Seg_1[j]
            Orig_2 = Seg_1[j + 1]
            Orig_3 = Seg_1[j + 2]
            Orig_4 = Seg_1[j + 3]
            Orig_5 = Seg_1[j + 4]
            plt.suptitle('Segmented Images from Dataset', fontsize=20)

            plt.subplot(3, 3, 1).axis('off')
            plt.imshow(GT)
            plt.title('Ground Truth', fontsize=10)

            plt.subplot(3, 3, 2).axis('off')
            plt.imshow(Orig_1)
            plt.title('Unet', fontsize=10)

            plt.subplot(3, 3, 3).axis('off')
            plt.imshow(Orig_2)
            plt.title('Unet3+', fontsize=10)

            plt.subplot(3, 3, 5).axis('off')
            plt.imshow(Orig)
            plt.title('Original', fontsize=10)

            plt.subplot(3, 3, 7).axis('off')
            plt.imshow(Orig_3)
            plt.title('TransUnet ', fontsize=10)

            plt.subplot(3, 3, 8).axis('off')
            plt.imshow(Orig_4)
            plt.title('A-3D-TDUNet', fontsize=10)

            plt.subplot(3, 3, 9).axis('off')
            plt.imshow(Orig_5)
            plt.title('PROPOSED', fontsize=10)

            path = "./Results/img_res/%s_image.png" % (str(i + 1))
            plt.savefig(path)
            plt.show()

            cv.imwrite('./Results/img_res/Original_image_' + str(i + 1) + '.png', Orig)
            cv.imwrite('./Results/img_res/segm_img_Ground_Truth_' + str(i + 1) + '.png', GT)
            cv.imwrite('./Results/img_res/segm_img_Unet_' + str(i + 1) + '.png', Orig_1)
            cv.imwrite('./Results/img_res/segm_img_Res-Unet_' + str(i + 1) + '.png', Orig_2)
            cv.imwrite('./Results/img_res/segm_img_Trans-Unet_' + str(i + 1) + '.png', Orig_3)
            cv.imwrite('./Results/img_res/segm_img_A-3D-TDUNet_' + str(i + 1) + '.png', Orig_4)
            cv.imwrite('./Results/img_res/segm_img_PROPOSED_' + str(i + 1) + '.png', Orig_5)


def PR_Curve():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    act = np.asarray(Actual).argmax(axis=1)
    pred = np.asarray(Predict).argmax(axis=1)
    precision, recall, thresholds = precision_recall_curve(Actual[:, 0], Predict[:, 0])
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./Results/PR Curve.png")
    plt.show()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    # cm = confusion_matrix(np.asarray(Actual), np.asarray(Predict))
    cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))
    Classes = ['Normal', 'Abnormal']
    rot = 0
    fontsizes = 12
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d', text_kw={'fontsize': 12})
    ax.set_xlabel('Predicted labels', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual labels', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
    ax.set_yticklabels(Classes, fontsize=fontsizes)
    plt.tight_layout()
    path = "./Results/Confusion_matrix.png"
    plt.savefig(path)
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


if __name__ == '__main__':
    plot_conv()
    ROC_curve()
    PR_Curve()
    Plot_Confusion()
    Plot_BatchSize()
    plot_results_Seg()
    Image_segment_comparision()
