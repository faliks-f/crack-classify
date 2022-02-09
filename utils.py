class Result:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.correct = 0
        self.total = 0

    def clear(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.correct = 0
        self.total = 0

    def update(self, output, label):
        for j in range(len(output)):
            if output[j] > 0.5 and label[j] == 1:
                self.TP += 1
            elif output[j] > 0.5 and label[j] == 0:
                self.FP += 1
            elif output[j] < 0.5 and label[j] == 1:
                self.FN += 1
            if abs(output[j] - label[j]) < 0.5:
                self.correct += 1
            self.total += 1

    def get_result(self):
        precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) != 0 else 0
        recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) != 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = self.correct / self.total
        return precision, recall, F1, accuracy
