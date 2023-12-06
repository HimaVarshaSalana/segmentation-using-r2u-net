import torch

# SR : Segmentation Result
# GT : Ground Truth


def get_accuracy(SR, GT, threshold=0.03):
    #print(SR)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    #mask = GT == 1
    #print(SR[mask])

    #SR = SR > threshold
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.03):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = torch.sum((SR == 1) & (GT == 1)).item()
    FN = torch.sum((SR == 1) & (GT == 1)).item()
    # print(TP,FN)
    SE = TP/(TP + FN + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.03):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = torch.sum((SR == 0) & (GT == 0)).item()
    FP = torch.sum((SR == 1) & (GT == 0)).item()

    SP = float(TN/(TN+FP + 1e-6))

    return SP


def get_precision(SR, GT, threshold=0.03):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = torch.sum((SR == 1) & (GT == 1)).item()
    FP = torch.sum((SR == 1) & (GT == 0)).item()

    PC = float(TP/(TP+FP + 1e-6))

    return PC


def get_F1(SR, GT, threshold=0.03):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.03):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)

    intersection = torch.sum((SR.bool() & GT.bool()).float())
    union = torch.sum((SR.bool() | GT.bool()).float())
    jaccard = intersection / (union + 1e-6)
    return jaccard

    


def get_DC(SR, GT, threshold=0.03):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    intersection = torch.sum((SR.bool() & GT.bool()).float())
    dice = (2.0 * intersection) / (torch.sum(SR.bool().float()) + torch.sum(GT.bool().float()) + 1e-6)
    return dice
