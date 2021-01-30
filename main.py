"""
Name : main.py
Author : Chayma Zatout
Contact : https://github.com/ChaymaZatout
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
    print("Creating bassar __time__: "+str(time.time()-start)+" s.")
    start = time.time()
    vis.add_geometry(basisr.create_base([255, 255, 255]))
    print("Adding bassar as geometry __time__: " + str(time.time()-start) + " s.")

    # cylinders:
    start = time.time()
    basisr.map_segments(segments)
    print("Mapping segments __time__: " + str(time.time()-start) + " s.")
    start = time.time()
    basisr.update_pinsClolor()
    print("Updating colors __time__: " + str(time.time()-start) + " s.")
    start = time.time()
    pins = basisr.create_pins()
    print("Creating pins __time__: " + str(time.time()-start) + " s.")
    start = time.time()
    for p in pins:
        vis.add_geometry(p)
    print("Adding geometry __time__: " + str(time.time()-start) + " s.")
    # visualize:
    vis.run()
    vis.destroy_window()
