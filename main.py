"""
Name : main.py
Author : Chayma Zatout
Contact : _
Time    : 09/01/21 05:49 م
Desc:
"""
import numpy as np
from simulator import BASISR
import open3d as o3d
import time

if __name__ == '__main__':
    # Read data:
    segments = np.load("_in/segments.npy", allow_pickle=True)

    # init visualizer:
    vis = o3d.visualization.Visualizer()
    vis.create_window("BASSAR")
    vis.get_render_option().background_color = [0.75, 0.75, 0.75]
    vis.get_render_option().mesh_show_back_face = True

    # create bassar:
    small_base = 50
    base = 250
    height = 183.52

    start = time.time()
    basisr = BASISR(small_base, base, height)
    vis.add_geometry(basisr.create_base([255, 255, 255]))

    # cylinders:
    basisr.map_segments(segments)
    basisr.update_pinsClolor()
    pins = basisr.create_pins()
    for p in pins:
        vis.add_geometry(p)
    print("__time__="+str(time.time()-start)+"s.")
    # visualize:
    vis.run()
    vis.destroy_window()
