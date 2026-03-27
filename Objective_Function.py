import numpy as np
from Global_Vars import Global_vars
from Model_ADNet_DenseNet import Model_ADNet_DenseNet


def objfun_cls(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_ADNet_DenseNet(Feat, Tar, sol=sol)
            Fitn[i] = (1 / Eval[4] + Eval[6]) + Eval[8] + Eval[7] + Eval[10]  # Accuracy + Precision + FNR + FPR + FDR
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, predict = Model_ADNet_DenseNet(Feat, Tar, sol=sol)
        Fitn = (1 / Eval[4] + Eval[6]) + Eval[8] + Eval[7] + Eval[10]  # Accuracy + Precision + FNR + FPR + FDR
        return Fitn
