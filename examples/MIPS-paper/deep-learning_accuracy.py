import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import zarr
from scipy import stats
import umetrics 
from skimage.metrics import variation_of_information


# ---------
# Functions
# ---------


def get_accuracy_metrics(
    slices, 
    gt_data,
    model_result,
    VI: bool = True, 
    AP: bool = True, 
    ND: bool = True,
    out_path = None,
    exclude_chunks: int = 10, 
    ):
    '''
    Parameters:
    slices: list of tupel of 4 x slice 
        Slices denoting where the image chunks to be taken as single units of
        network output end and begin.
    gt_data: napari.types.LabelsData or int array like
        Stack of N x 3D validation ground truth (i.e., 4D with N,z,y,x)
    model_result: napari.types.LabelsData or int array like
        Stack of N x 3D model derived segmentation (i.e., 4D with N,z,y,x)
    VI: bool
        Should we find variation of information scores
    AP: bool
        Should we find average precision scores (for IoU 0.6-0.9)
    ND: bool
        SHould we find the number of objects difference from 
    '''
    scores = {'VI: GT | Output' : [], 'VI: Output | GT' : [], 'Count difference' : []}
    IoU_dict = generate_IoU_dict()
    scores.update(IoU_dict)
    for s_, c_ in slices:
        if isinstance(gt_data, np.ndarray):
            gt = gt_data(s_)
        else:
            gt = gt_data.data[s_]
        gt = np.squeeze(gt)[c_]
        n_objects = np.unique(gt).size
        if n_objects > exclude_chunks + 1:
            mr = model_result.data[s_]
            mr = np.squeeze(mr)[c_]
            #print('n_objects', n_objects)
            if VI:
                vi = variation_of_information(gt, mr)
                scores['VI: GT | Output'].append(vi[0])
                scores['VI: Output | GT'].append(vi[1])
            if AP:
                generate_IoU_data(gt, mr, scores)
            if ND:
                n_mr = np.unique(mr).size
                #print('n_mr', n_mr, np.unique(mr), mr.shape, mr.dtype)
                nd = n_mr - n_objects
                nd = nd / n_objects # as a proportion might be more informative
                scores['Count difference'].append(nd)
    lens = {key : len(scores[key]) for key in scores.keys()}
    to_keep = [key for key in scores.keys() if lens[key] > 1]
    new_scores = {key : scores[key] for key in to_keep}
    new_scores = pd.DataFrame(new_scores)
    statistics = single_sample_stats(new_scores, to_keep, model_result.name)
    new_scores['model_name'] = [model_result.name, ] * len(new_scores)
    if out_path is not None:
        new_scores.to_csv(out_path)
        p = Path(out_path)
        stat_path = os.path.join(p.parents[0], p.stem + '_stats.csv')
        statistics.to_csv(stat_path)
    ap_scores = None
    if AP:
        if out_path is not None:
            save_dir = Path(out_path).parents[0]
            name = Path(out_path).stem
            suffix = 'AP-scores'
        ap_scores = generate_ap_scores(new_scores, name, save_dir, suffix)
        ap_scores['model_name'] = [model_result.name, ] * len(ap_scores)
    return (new_scores, ap_scores), statistics


def single_sample_stats(df, columns, name):
    results = {}
    alpha = 0.95
    for c in columns:
        sample_mean = np.mean(df[c].values)
        sample_sem = stats.sem(df[c].values)
        degrees_freedom = df[c].values.size - 1
        CI = stats.t.interval(alpha, degrees_freedom, sample_mean, sample_sem)
        n = str(c) + '_'
        results[n + 'mean'] = sample_mean
        results[n + 'sem'] = sample_sem
        results[n + '95pcntCI_2-5pcnt'] = CI[0]
        results[n + '95pcntCI_97-5pcnt'] = CI[1]
    results = pd.DataFrame(results)
    results['model_name'] = [name, ] * len(df)
    return results



def metrics_for_stack(directory, name, seg, gt):
    assert seg.shape[0] == gt.shape[0]
    IoU_dict = generate_IoU_dict()
    for i in range(seg.shape[0]):
        seg_i = seg[i].compute()
        gt_i = gt[i].compute()
        generate_IoU_data(gt_i, seg_i, IoU_dict)
    df = save_data(IoU_dict, name, directory, 'metrics')
    ap = generate_ap_scores(df, name, directory)
    return df, ap


def calc_ap(result):
        denominator = result.n_true_positives + result.n_false_negatives + result.n_false_positives
        return result.n_true_positives / denominator


def generate_IoU_dict(thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    IoU_dict = {}
    IoU_dict['n_predicted'] = []
    IoU_dict['n_true'] = []
    IoU_dict['n_diff'] = []
    for t in thresholds:
        n = f't{t}_true_positives'
        IoU_dict[n] = []
        n = f't{t}_false_positives'
        IoU_dict[n] = []
        n = f't{t}_false_negatives'
        IoU_dict[n] = []
        n = f't{t}_IoU'
        IoU_dict[n] = []
        n = f't{t}_Jaccard'
        IoU_dict[n] = []
        n = f't{t}_pixel_identity'
        IoU_dict[n] = []
        n = f't{t}_localization_error'
        IoU_dict[n] = []
        n = f't{t}_per_image_average_precision'
        IoU_dict[n] = []
    return IoU_dict


def generate_IoU_data(gt, seg, IoU_dict, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    for t in thresholds:
        result = umetrics.calculate(gt, seg, strict=True, iou_threshold=t)
        n = f't{t}_true_positives'
        IoU_dict[n].append(result.n_true_positives) 
        n = f't{t}_false_positives'
        IoU_dict[n].append(result.n_false_positives) 
        n = f't{t}_false_negatives'
        IoU_dict[n].append(result.n_false_negatives) 
        n = f't{t}_IoU'
        IoU_dict[n].append(result.results.IoU) 
        n = f't{t}_Jaccard'
        IoU_dict[n].append(result.results.Jaccard) 
        n = f't{t}_pixel_identity'
        IoU_dict[n].append(result.results.pixel_identity) 
        n = f't{t}_localization_error'
        IoU_dict[n].append(result.results.localization_error) 
        n = f't{t}_per_image_average_precision'
        IoU_dict[n].append(calc_ap(result))
        if t == thresholds[0]:
            IoU_dict['n_predicted'].append(result.n_pred_labels)
            IoU_dict['n_true'].append(result.n_true_labels)
            IoU_dict['n_diff'].append(result.n_true_labels - result.n_pred_labels)


def save_data(data_dict, name, directory, suffix):
    df = pd.DataFrame(data_dict)
    n = name + '_' + suffix +'.csv'
    p = os.path.join(directory, n)
    df.to_csv(p)
    return df


def generate_ap_scores(df, name, directory, suffix, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    ap_scores = {'average_precision' : [], 
                 'threshold': []}
    for t in thresholds:
        ap_scores['threshold'].append(t)
        n = f't{t}_true_positives'
        true_positives = df[n].sum()
        n = f't{t}_false_positives'
        false_positives = df[n].sum()
        n = f't{t}_false_negatives'
        false_negatives = df[n].sum()
        ap = true_positives / (true_positives + false_negatives + false_positives)
        ap_scores['average_precision'].append(ap)
    print(ap_scores)
    if directory is not None:
        ap_scores = save_data(ap_scores, name, directory, suffix)
    return ap_scores


# ---------
# Execution
# ---------
import os
# paths
d = '/Users/abigailmcgovern/Data/platelet-analysis/DL-validation/2306'
seg_d = os.path.join(d, 'segmentations')
gt_d = os.path.join(d, 'ground_truths')
# file lists
seg_ps = os.listdir(seg_d)
gt_ps = os.listdir(gt_d)
# read files
segs = [np.array(zarr.open(p)) for p in seg_ps]
gts = [np.array(zarr.open(p)) for p in gt_ps]
# convert to format
segs = np.stack(segs)
gts = np.stack(segs)





