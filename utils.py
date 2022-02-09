def cal_res(TP, FP, FN, correct, total):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = correct / total
    return precision, recall, F1, accuracy


def judge(output, label, TP, FP, FN, correct, total):
    for j in range(len(output)):
        if output[j] > 0.5 and label[j] == 1:
            TP += 1
        elif output[j] > 0.5 and label[j] == 0:
            FP += 1
        elif output[j] < 0.5 and label[j] == 1:
            FN += 1
        if abs(output[j] - label[j]) < 0.5:
            correct += 1
        total += 1
    return TP, FP, FN, correct, total
