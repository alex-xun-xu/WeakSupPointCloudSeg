import os
import sys
import numpy as np
import tensorflow as tf

###### Define Class for Evaluation
class ShapeNetEval():

    #### Data
    AP = 0

    #### Methods
    def EvalIoU(self,pred,seg_gt,iou_oids):

        ## Evaluate IoU
        mask = np.int32(pred == seg_gt)

        total_iou = 0.0
        iou_log = ''
        # iou_oids = np.unique(seg_gt)
        for oid in iou_oids:
            n_pred = np.sum(pred == oid)
            n_gt = np.sum(seg_gt == oid)
            n_intersect = np.sum(np.int32(seg_gt == oid) * mask)
            n_union = n_pred + n_gt - n_intersect
            iou_log += '_' + str(n_pred) + '_' + str(n_gt) + '_' + str(n_intersect) + '_' + str(n_union) + '_'
            if n_union == 0:
                total_iou += 1
                iou_log += '_1\n'
            else:
                total_iou += n_intersect * 1.0 / n_union
                iou_log += '_' + str(n_intersect * 1.0 / n_union) + '\n'

        avg_iou = total_iou / len(iou_oids)

        return avg_iou
