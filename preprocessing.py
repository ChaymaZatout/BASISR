"""
Name : preprocessing.py
Author : Chayma Zatout
Contact : _
Time    : 26/01/21 02:57 م
Desc:
"""


def map_point(point):
    return [25 * point[0] / 435, 0.0575 * point[1], -183.52 * (point[2] - 800) / 3200]


def map_pcd(pcd):
    return [map_point(p) for p in pcd]
