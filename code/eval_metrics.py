import torch

class F1ScoreSeq2Seq():
    """F1 score for evaluation of Seq2Seq models."""
    def __init__(self):
        """Initialize the F1 score metric."""
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def calculate_precision(self):
        """
        Calculate precision. Precision is calculated as tp / (tp + fp).

        :return: precision
        """
        if self.tp + self.fp == 0:
            return 0.0
        precision = self.tp / (self.tp + self.fp)
        return precision

    def calculate_recall(self):
        """
        Calculate recall. Recall is calculated as tp / (tp + fn).

        :return: recall
        """
        if self.tp + self.fn == 0:
            return 0.0
        recall = self.tp / (self.tp + self.fn)
        return recall

    def compute(self):
        """
        Calculate F1 score. F1 score is calculated as 2 * (precision * recall) / (precision + recall).

        :return: F1 score
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def update(
            self,
            predictions,
            labels,
    ):
        """
        Update the metric with given predictions and labels.

        :param predictions: predictions
        :param labels: labels
        :return: None
        """
        for prediction in predictions:
            if prediction in labels:
                self.tp += 1
            else:
                self.fp += 1

        for label in labels:
            if label not in predictions:
                self.fn += 1
                
    def eval(self, predictions_list, labels_list):
        if len(predictions_list) != len(labels_list):
            raise ValueError("Predictions and labels must have the same length.")
        for predictions, labels in zip(predictions_list, labels_list):
            self.update(predictions, labels)
        return self.compute(), self.calculate_recall(), self.calculate_precision() # F1, Recall, Precision
    
    def eval_mae(self, predictions_list, labels_list, predictions_sis_list, labels_sis_list):
        if len(predictions_list) != len(labels_list):
            raise ValueError("Predictions and labels must have the same length.")
        
        mae_list = []
        for predictions, labels, predictions_sis, labels_sis in zip(predictions_list, labels_list, predictions_sis_list, labels_sis_list):
            for pred_index in range(len(predictions)):
                if predictions[pred_index] in labels:
                    label_index = labels.index(predictions[pred_index])
                    
                    mae = abs(float(predictions_sis[pred_index][-1]) - float(labels_sis[label_index][-1]))
                    mae_list.append(mae)
        return mae_list
            
            
