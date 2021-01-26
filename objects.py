"""
Name : objects.py
Author : Chayma Zatout
Contact : _
Time    : 26/01/21 02:51 م
Desc:
"""

import numpy as np


class Segment:

    def __init__(self, pcd, ymin, name=None):
        self.pcd = pcd
        self.classe = name
        self.ymin = ymin

    def centroid(self):
        xs = np.array([p[0] for p in self.pcd])
        ys = np.array([p[1] for p in self.pcd])
        zs = np.array([p[2] for p in self.pcd])

        return [(np.percentile(xs, 10) + np.percentile(xs, 90)) / 2,
                np.percentile(ys, 90),
                (np.percentile(zs, 10) + np.percentile(zs, 90)) / 2]

    def height(self):
        ys = np.array([p[1] for p in self.pcd])
        return abs(np.percentile(ys, 95) - self.ymin)

    def height_class(self, b25, b75):
        height = self.height()
        if height < b25:
            return 1
        elif height < b75:
            return 2
        else:
            return 3
