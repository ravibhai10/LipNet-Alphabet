"""Heuristic agreement of temporal feature windows; no extra encoder pass.

Windows share encoder context and are correlated, not independent examples.
Their agreement is a rejection rule, not calibrated correctness or OOV proof.
"""
import numpy as np
from src.video_training import resample_temporal
from .pretrained import sequence_probabilities


def temporal_consensus(head, features, policy):
    if features.shape != (29,768) or not np.isfinite(features).all():
        raise ValueError('Finite 29-frame visual features required for consensus.')
    views=[]
    for start,end in ((0,25),(2,27),(4,29)):
        p=sequence_probabilities(head,resample_temporal(features[start:end],29))
        order=np.argsort(-p);confidence=float(p[order[0]]);margin=float(p[order[0]]-p[order[1]])
        views.append({'bounds':[start,end],'candidate':head['vocabulary'][int(order[0])],
                      'confidence':confidence,'margin':margin,
                      'accepted':confidence>=policy['threshold'] and margin>=policy['margin']})
    return {'confirmed':all(v['accepted'] for v in views) and len({v['candidate'] for v in views})==1,
            'views':views,'method':'three_overlapping_feature_windows',
            'independent_examples':False,'calibrated':False}
