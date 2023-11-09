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
import itertools


# ---------
# Functions
# ---------

def make_chunks(arr_shape, chunk_shape, margin):
    ndim = len(arr_shape)
    if type(margin) == int:
        margin = [margin] * ndim
    starts = []
    crops = []
    for dim in range(ndim):
        arr = arr_shape[dim]
        chk = chunk_shape[dim]
        mrg = margin[dim]
        start = np.arange(0, arr - 2*mrg, chk - 2*mrg)
        start[-1] = arr - chk
        if len(start) > 1 and start[-1] == start[-2]:
            # remove duplicates in case last step is perfect
            start = start[:-1]
        starts.append(start)
        crop = np.array([(mrg, chk - mrg),] * len(start))  # yapf: disable
        crop[0, 0] = 0
        crop[-1, 0] = chk - (arr - np.sum(crop[:-1, 1] - crop[:-1, 0]))
        crop[-1, 1] = chk
        crops.append(crop)
    chunk_starts = list(itertools.product(*starts))
    chunk_crops = list(itertools.product(*crops))
    return chunk_starts, chunk_crops


def get_slices_from_chunks(arr_shape, chunk_size, margin):
    if len(arr_shape) <= 3:
        ts = range(1)
        fshape = arr_shape
    else:
        ts = range(arr_shape[0])
        fshape = arr_shape[1:]
    chunk_starts, chunk_crops = make_chunks(fshape, chunk_size, margin)
    slices = []
    for t in ts:
        for start, crop in list(zip(chunk_starts, chunk_crops)):
            sl = (slice(t, t+1), ) + tuple(
                    slice(start0, start0 + step)
                    for start0, step in zip(start, chunk_size)
                    )
            cr = tuple(slice(i, j) for i, j in crop)
            slices.append((sl, cr)) # useage: 4d_labels[sl][cr]
    return slices


def get_accuracy_metrics(
    slices, 
    gt_data,
    model_result,
    VI: bool = True, 
    AP: bool = True, 
    ND: bool = True,
    out_path = None,
    exclude_chunks: int = 10, 
    is3D=True
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
    scores = {
        'VI: GT | Output' : [], 
        'VI: Output | GT' : [], 
        'Number objects (GT)' : [], 
        'Number objects (model)' : [], 
        'Count difference' : [], 
        'Count difference (%)' : [], 
        }
    IoU_dict = generate_IoU_dict()
    scores.update(IoU_dict)
    for s_, c_ in slices:
        if is3D:
            s_ = s_[1:]
        if isinstance(gt_data, np.ndarray):
            gt = gt_data[s_]
        else:
            gt = gt_data.data[s_]
        gt = np.squeeze(gt)[c_]
        n_objects = np.unique(gt).size
        if n_objects > exclude_chunks + 1:
            if isinstance(model_result, np.ndarray):
                mr = model_result[s_]
            else:
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
                ndp = nd / n_objects * 100 # as a percent might be more informative
                scores['Count difference (%)'].append(ndp)
                scores['Number objects (GT)'].append(n_objects)
                scores['Number objects (model)'].append(n_mr)
                scores['Count difference'].append(nd)
    lens = {key : len(scores[key]) for key in scores.keys()}
    to_keep = [key for key in scores.keys() if lens[key] > 1]
    new_scores = {key : scores[key] for key in to_keep}
    new_scores = pd.DataFrame(new_scores)
    if isinstance(model_result, np.ndarray):
        model_name = 'segmentation'
    else:
        model_name = model_result.name
    statistics = single_sample_stats(new_scores, to_keep, model_name)
    new_scores['model_name'] = [model_name, ] * len(new_scores)
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
        else:
            name = 'segmentation'
            save_dir = None
        suffix = 'AP-scores'
        ap_scores = generate_ap_scores(new_scores, name, save_dir, suffix)
        ap_scores['model_name'] = [model_name, ] * len(ap_scores)
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
        results[n + 'mean'] = [sample_mean, ]
        results[n + 'sem'] = [sample_sem, ]
        results[n + '95pcntCI_2-5pcnt'] = [CI[0], ]
        results[n + '95pcntCI_97-5pcnt'] = [CI[1], ]
    results = pd.DataFrame(results)
    results['model_name'] = name
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
    ap_scores = pd.DataFrame(ap_scores)
    return ap_scores


def load_input_data(seg_d, gt_d):
    seg_ps = [os.path.join(seg_d, f) for f in os.listdir(seg_d) if f.endswith('.zarr')]
    gt_ps = [os.path.join(gt_d, f) for f in os.listdir(gt_d) if f.endswith('.zarr')]
    val_dict = {}
    for p in seg_ps: #_labels
        seg = np.array(zarr.open(p))
        name = Path(p).stem[:-7]
        val_dict[name] = {}
        val_dict[name]['segmentation'] = seg
        gtps = [p for p in gt_ps if p.find(name) != -1]
        gt_names = [Path(p).stem[-2:] for p in gtps]
        gts = [np.array(zarr.open(p)) for p in gtps]
        val_dict[name]['GT_names'] = gt_names
        val_dict[name]['GTs'] = gts
    return val_dict


def get_data_sheets(seg_d, gt_d, d, prefix):
    val_dict = load_input_data(seg_d, gt_d)
    scores_all = []
    stats_all = []
    ap_all = []
    slices = get_slices_from_chunks((33, 512, 512), (10, 265, 256), (1, 64, 64))
    for n in val_dict.keys():
        seg = val_dict[n]['segmentation']
        for i, gt_n in enumerate(val_dict[n]['GT_names']):
            gt = val_dict[n]['GTs'][i]
            scores, statistics = get_accuracy_metrics(slices, gt, seg)
            new_scores, ap_scores = scores
            # add column for source info
            statistics['file'] = n
            statistics['person'] = gt_n
            stats_all.append(statistics)
            new_scores['file'] = n
            new_scores['person'] = gt_n
            scores_all.append(new_scores)
            ap_scores['file'] = n
            ap_scores['person'] = gt_n
            ap_all.append(ap_scores)
    # concat
    scores_all = pd.concat(scores_all).reset_index(drop=True)
    stats_all = pd.concat(stats_all).reset_index(drop=True)  
    ap_all = pd.concat(ap_all).reset_index(drop=True)  
    # save data
    scores_p = os.path.join(d, f'{prefix}_scores.csv')
    scores_all.to_csv(scores_p)
    ap_p = os.path.join(d, f'{prefix}_ap_scores.csv')
    ap_all.to_csv(ap_p)
    stats_p = os.path.join(d, f'{prefix}_stats.csv')
    stats_all.to_csv(stats_p)

        


# ---------
# Execution
# ---------
import os
from pathlib import Path
# paths
d = '/Users/abigailmcgovern/Data/platelet-analysis/DL-validation/2306'
seg_d = os.path.join(d, 'segmentations')
gt_d = os.path.join(d, 'ground_truths')
get_data_sheets(seg_d, gt_d, d, '230630_DLvalidation')

