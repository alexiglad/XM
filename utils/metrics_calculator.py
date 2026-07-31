import torch
import torchmetrics
import nltk
import string
from typing import List

def get_torchmetrics(metric, metrics_average_type, num_classes, metrics_task):
    if 'accuracy' in metric:
        return torchmetrics.Accuracy(average=metrics_average_type, num_classes=num_classes, task=metrics_task)
    elif 'f1_score' in metric:
        return torchmetrics.F1Score(average=metrics_average_type, num_classes=num_classes, task=metrics_task)
    elif 'precision' in metric:
        return torchmetrics.Precision(average=metrics_average_type, num_classes=num_classes, task=metrics_task)
    elif 'recall' in metric:
        return torchmetrics.Recall(average=metrics_average_type, num_classes=num_classes, task=metrics_task)
    else:
        raise ValueError(f"metric {metric} unimplemented")


def calculate_em(predictions: List[str], references: List[str]) -> float:
    '''
    Method for calculating the Exact Match (EM) score for Question Answering / Text Generation tasks
    '''
    assert len(predictions) > 0, "Predictions list cannot be empty"
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    total = len(predictions)
    correct = sum(
        normalize_text(pred) == normalize_text(ref)
        for pred, ref in zip(predictions, references)
    )
    return correct / total


def calculate_f1_score(predictions: List[str], references: List[str]) -> float:
    '''
    Method for calculating the F1 score for Question Answering / Text Generation tasks
    '''
    assert len(predictions) > 0, "Predictions list cannot be empty"
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    total_f1 = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(normalize_text(pred))
        ref_tokens = nltk.word_tokenize(normalize_text(ref))

        pred_counter = {}
        ref_counter = {}
        for token in pred_tokens:
            pred_counter[token] = pred_counter.get(token, 0) + 1
        for token in ref_tokens:
            ref_counter[token] = ref_counter.get(token, 0) + 1

        matches = 0
        for token in pred_counter:
            if token in ref_counter:
                matches += min(pred_counter[token], ref_counter[token])

        precision = matches / len(pred_tokens) if pred_tokens else 0
        recall = matches / len(ref_tokens) if ref_tokens else 0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        total_f1 += f1

    return total_f1 / len(predictions)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    text = ' '.join(text.split())
    return text