import torch
import numpy as np

def get_ious(boxes1, boxes2):
    '''
    Parameters:
        boxes1: (845, 4) corner
        boxes2: (n_obj, 4) corner

    Returns:
        IoUs: (n1, n2)
    '''

    # Intersection
    top_left_xy = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2].unsqueeze(0)) # (n1, n2, 2)
    bottom_right_xy = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:].unsqueeze(0)) # (n1, n2, 2)

    inter_wh = torch.clamp(bottom_right_xy - top_left_xy, min=0) # (n1, n2, 2)
    intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1] # (n1, n2)

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # (n1, )
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # (n2, )

    union = boxes1_area.unsqueeze(1) + boxes2_area.unsqueeze(0) - intersection + 1e-6

    return intersection / union # (n1, n2)
