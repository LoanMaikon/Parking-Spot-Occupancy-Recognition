import sys
import os
import numpy as np
import json
from statistics import mean
from datetime import datetime

def get_accuracy(predictions, targets, image_paths):
    predictions_and_targets_per_subset = _get_info_per_subset(predictions, targets, image_paths)

    accuracies = []
    samples = []
    for subset, info in predictions_and_targets_per_subset.items():
        correct = sum([1 if p == t else 0 for p, t in zip(info['predictions'], info['targets'])])
        total = len(info['targets'])
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)
        samples.append(total)

    return sum([acc * (s / sum(samples)) for acc, s in zip(accuracies, samples)]) if sum(samples) > 0 else 0

def get_macro_f1(predictions, targets, image_paths):
    predictions_and_targets_per_subset = _get_info_per_subset(predictions, targets, image_paths)

    f1_scores = []
    samples = []
    for subset, info in predictions_and_targets_per_subset.items():
        preds = info['predictions']
        targs = info['targets']

        tp_class0 = sum([1 if p == 0 and t == 0 else 0 for p, t in zip(preds, targs)])
        tp_class1 = sum([1 if p == 1 and t == 1 else 0 for p, t in zip(preds, targs)])
        fp_class0 = sum([1 if p == 0 and t == 1 else 0 for p, t in zip(preds, targs)])
        fp_class1 = sum([1 if p == 1 and t == 0 else 0 for p, t in zip(preds, targs)])
        fn_class0 = sum([1 if p == 1 and t == 0 else 0 for p, t in zip(preds, targs)])
        fn_class1 = sum([1 if p == 0 and t == 1 else 0 for p, t in zip(preds, targs)])

        precision_class0 = tp_class0 / (tp_class0 + fp_class0) if (tp_class0 + fp_class0) > 0 else 0
        precision_class1 = tp_class1 / (tp_class1 + fp_class1) if (tp_class1 + fp_class1) > 0 else 0

        recall_class0 = tp_class0 / (tp_class0 + fn_class0) if (tp_class0 + fn_class0) > 0 else 0
        recall_class1 = tp_class1 / (tp_class1 + fn_class1) if (tp_class1 + fn_class1) > 0 else 0

        f1_class0 = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0) if (precision_class0 + recall_class0) > 0 else 0
        f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1) if (precision_class1 + recall_class1) > 0 else 0

        f1_scores.append((f1_class0 + f1_class1) / 2)
        samples.append(len(targs))

    total_samples = sum(samples)
    return sum([f1 * (s / total_samples) for f1, s in zip(f1_scores, samples)]) if total_samples > 0 else 0

def _get_info_per_subset(predictions, targets, image_paths, threshold=0.5):
    predictions_and_targets_per_subset = {}

    for prediction, target, image_path in zip(predictions, targets, image_paths):
        subset = _get_subset(image_path)

        if subset not in predictions_and_targets_per_subset:
            predictions_and_targets_per_subset[subset] = {'predictions': [], 'targets': [], 'image_paths': []}

        predictions_and_targets_per_subset[subset]['predictions'].append(1 if prediction >= threshold else 0)
        predictions_and_targets_per_subset[subset]['targets'].append(target)
        predictions_and_targets_per_subset[subset]['image_paths'].append(image_path)

    return predictions_and_targets_per_subset

def _get_subset(image_path):
    return image_path.split('/')[-4]

def _get_class(image_path):
    class_name = image_path.split('/')[-2].lower()

    return 0 if class_name == 'empty' else 1
