# sort.py - Simple Online Realtime Tracking
# Based on the SORT algorithm by Alex Bewley et al.
# https://github.com/abewley/sort

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bb_test: [x1, y1, x2, y2]
        bb_gt: [x1, y1, x2, y2]
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    
    if area_test <= 0 or area_gt <= 0:
        return 0.0
    
    intersection = w * h
    union = area_test + area_gt - intersection
    
    return intersection / union


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10  # Measurement uncertainty
        self.kf.P[4:, 4:] *= 1000  # Give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Convert [x1, y1, x2, y2] to [x, y, s, r] where x,y is the center,
        # s is the scale (area) and r is the aspect ratio
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Store the original bbox for visualization
        self.bbox = bbox
        
        # Add attributes for vehicle tracking
        self.class_id = None
        self.class_name = None
        self.detection_index = None  # To link back to original detection
        self.counted = False
        
    def convert_bbox_to_z(self, bbox):
        """
        Convert bounding box format from [x1, y1, x2, y2] to [x, y, s, r]
        where x,y is the center, s is the scale (area) and r is the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    # area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def convert_x_to_bbox(self, x, score=None):
        """
        Convert state vector [x, y, s, r] to bounding box [x1, y1, x2, y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        
        if score is None:
            return np.array([
                x[0] - w/2.,
                x[1] - h/2.,
                x[0] + w/2.,
                x[1] + h/2.
            ]).reshape((1, 4))
        else:
            return np.array([
                x[0] - w/2.,
                x[1] - h/2.,
                x[0] + w/2.,
                x[1] + h/2.,
                score
            ]).reshape((1, 5))
    
    def update(self, bbox):
        """
        Update the state vector with observed bbox
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Store current bbox
        self.bbox = bbox
        
        # Update Kalman filter
        self.kf.update(self.convert_bbox_to_z(bbox))
    
    def predict(self):
        """
        Advance the state vector and return the predicted bounding box
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        
        # Update bbox with latest prediction
        predicted_bbox = self.convert_x_to_bbox(self.kf.x)[0]
        self.bbox = predicted_bbox.astype(int)
        
        return self.history[-1]
    
    def get_state(self):
        """
        Return the current bounding box estimate
        """
        return self.convert_x_to_bbox(self.kf.x)[0]
    
    def get_velocity(self):
        """
        Return the current velocity estimate (pixels per frame)
        """
        return np.array([self.kf.x[4], self.kf.x[5]]).reshape(2)


class SORT:
    """
    SORT - Simple Online and Realtime Tracking
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Parameters:
            max_age - Maximum frames to keep alive a track without matching detections
            min_hits - Minimum hits needed to start a track
            iou_threshold - IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets):
        """
        Update tracked objects with new detections
        
        Args:
            dets - a numpy array of detection bounding boxes in format [x1, y1, x2, y2, score]
        
        Returns:
            a numpy array of tracked objects in format [x1, y1, x2, y2, id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Calculate IoU matrix between predicted boxes and detected boxes
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)
            
        # Return active trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        
        Returns:
            3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)
                
        # Use the Hungarian algorithm to find the best matches
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        
        # Find unmatched detections
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        # Find unmatched trackers
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
                
        # Filter out matches with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)