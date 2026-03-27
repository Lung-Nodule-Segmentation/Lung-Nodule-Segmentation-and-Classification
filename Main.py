import numpy as np
import os
import pydicom as dicom
from numpy import matlib
from Global_Vars import Global_vars
from Model_3D_Trans_DenseUnet_Plus_plus import Model_3D_Trans_DenseUnet_Plus_plus
from Model_ADNet_DenseNet import Model_ADNet_DenseNet
from Model_DenseNet import Model_DenseNet
from Model_LSTM import Model_LSTM
from Model_RAN import Model_RAN
from Model_Trans_Unet import Model_Trans_Unet
from Model_Trans_Unet_plus_plus import Model_Trans_Unet_plus_plus
from Model_UNET import Model_Unet
from Model_Unet3Plus import Model_Unet3plus
from Objective_Function import objfun_cls
from PROPOSED import PROPOSED
from Plot_results import *
from RSA import RSA
from RTH import RTH
from SCO import SCO
from TFMOA import TFMOA


# Read Dataset-UnitoChest
an = 0
if an == 1:
    Images = []
    GroundTruth = []
    Path = './Dataset/UnitoChest'
    sub_dir = os.listdir(Path)
    for i in range(len(sub_dir)):
        if sub_dir[i] == 'Images':
            sub_dir_ = os.listdir(Path + '/' + sub_dir[i])
            for j in range(len(sub_dir_)):  # For all Patients
                # Read Images
                sub_dir__ = os.listdir(Path + '/' + sub_dir[i] + '/' + sub_dir_[j] + '/exam_1/')
                image_name = []
                for k in range(len(sub_dir__)):  # For all Images
                    image_path = Path + '/' + sub_dir[i] + '/' + sub_dir_[j] + '/exam_1/' + sub_dir__[k]
                    ds = dicom.dcmread(image_path)
                    image = (ds.pixel_array / 13).astype('uint8')
                    image = cv.resize(image, [128, 128])
                    image_name.append(sub_dir__[k].split('.')[0] + '_mask.png')
                    Images.append(image)

                # Read Masks
                image_name = np.asarray(image_name)
                mask_sub_dir__ = np.asarray(os.listdir(Path + '/' + sub_dir[i + 1] + '/' + sub_dir_[j] + '/exam_1/'))
                for k in range(len(sub_dir__)):  # For all Images
                    if image_name[k] in mask_sub_dir__:
                        image_path = Path + '/' + sub_dir[i + 1] + '/' + sub_dir_[j] + '/exam_1/' + image_name[k]
                        image = cv.imread(image_path)
                        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                        image = cv.resize(image, [128, 128])
                    else:
                        image = np.zeros((128, 128)).astype('uint8')
                    GroundTruth.append(image)

    np.save('Images.npy', Images)
    np.save('GroundTruth.npy', GroundTruth)

# Generate Target
an = 0
if an == 1:
    Tar = []
    Ground_Truth = np.load('GroundTruth.npy', allow_pickle=True)
    for i in range(len(Ground_Truth)):
        print(i, len(Ground_Truth))
        image = Ground_Truth[i]
        result = image.astype('uint8')
        uniq = np.unique(result)
        if len(uniq) > 1:
            Tar.append(1)
        else:
            Tar.append(0)
    Uni = np.unique(Tar)
    uni = np.asarray(Uni)
    Target = np.zeros((len(Tar), len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(Tar == uni[j])
        Target[ind, j] = 1
    np.save('Target.npy', Target)

# Segmentation
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)  # Load the Data
    Target = np.load('GroundTruth.npy', allow_pickle=True)  # Load the ground truth
    Unet = Model_Unet(Images, Target)
    Unet3plus = Model_Unet3plus(Images, Target)
    Trans_Unet = Model_Trans_Unet(Images, Target)
    Trans_Unet_plus_plus = Model_Trans_Unet_plus_plus(Images, Target)
    Proposed = Model_3D_Trans_DenseUnet_Plus_plus(Images, Target)
    Seg = [Unet, Unet3plus, Trans_Unet, Trans_Unet_plus_plus, Proposed]
    np.save('Segmented_image.npy', Proposed)
    np.save('Seg_img.npy', Seg)

# optimization for Classification
an = 0
if an == 1:
    Feat = np.load('Segmented_image.npy', allow_pickle=True)  # Load the Selected features
    Target = np.load('Target.npy', allow_pickle=True)  # Load the Target
    Global_vars.Feat = Feat
    Global_vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, epoch in DenseNet, Hidden Neuron Count in LSTM
    xmin = matlib.repmat(np.asarray([5, 5, 5]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 255]), Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("RSA...")
    [bestfit1, fitness1, bestsol1, time1] = RSA(initsol, fname, xmin, xmax, Max_iter)  # RSA

    print("TFMOA...")
    [bestfit2, fitness2, bestsol2, time2] = TFMOA(initsol, fname, xmin, xmax, Max_iter)  # TFMOA

    print("SCO...")
    [bestfit3, fitness3, bestsol3, time3] = SCO(initsol, fname, xmin, xmax, Max_iter)  # SCO

    print("RTH...")
    [bestfit4, fitness4, bestsol4, time4] = RTH(initsol, fname, xmin, xmax, Max_iter)  # RTH

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
    np.save('Fitness.npy', fitness)
    np.save('BestSol_CLS.npy', BestSol_CLS)

# Classification by Varying Batch Size
an = 0
if an == 1:
    EVAL = []
    Feat = np.load('Segmented_image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    BestSol = np.load('BestSol_cls.npy', allow_pickle=True)
    Batchsize = [4, 8, 16, 32, 48, 64]
    for Bs in range(len(Batchsize)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 15))
        for j in range(BestSol.shape[0]):
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_ADNet_DenseNet(Feat, Target, Batchsize[Bs], sol)
        Eval[5, :], pred1 = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize[Bs])
        Eval[6, :], pred2 = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize[Bs])
        Eval[7, :], pred3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize[Bs])
        Eval[8, :], pred4 = Model_ADNet_DenseNet(Feat, Target, Batchsize[Bs])
        Eval[9, :], pred5 = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all_BS.npy', EVAL)  # Save the Eval all

plot_conv()
ROC_curve()
PR_Curve()
Plot_Confusion()
Plot_BatchSize()
plot_results_Seg()
Image_segment_comparision()
