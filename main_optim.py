"""
Name : main_optim.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 21/03/21 07:04 م
Desc:
"""
import numpy as np
from simulator import BASISR
from preprocessing import map_pcd
import open3d as o3d
import time


def get_indices(pcd, n_lines, n_cols, trX, trZ):
    i = int(n_cols / 2) + pcd[:, 0] / trX + (1 if n_cols % 2 == 0 else 0)
    j = n_lines + pcd[:, 2] / trZ
    return np.array(j, dtype=int), np.array(i, dtype=int)


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
    print("Creating bassar __time__: " + str(time.time() - start) + " s.")
    start = time.time()
    vis.add_geometry(basisr.create_base([255, 255, 255]))
    print("Adding bassar as geometry __time__: " + str(time.time() - start) + " s.")

    # add cylinders:
    start = time.time()
    for p in basisr.pins.flatten():
        if p is not None:
            vis.add_geometry(p)
    print("Adding pins __time__: " + str(time.time() - start) + " s.")

    # map segments:
    h = np.zeros((basisr.n_lines, basisr.n_cols))

    start = time.time()
    for s in segments:
        # compute height:
        pcd = np.array(s.pcd)
        pcd[:, 1] = pcd[:, 1] - s.ymin
        # map:
        h = basisr.compute_from_pcd(pcd, h, s.height_class(1200,1800))
    print(f"Mapping segments __time__: {time.time() - start} s.")

    start = time.time()
    basisr.update_pins(h)
    print(f"Updating __time__: {time.time() - start}")

    # show
    vis.run()
    vis.destroy_window()
