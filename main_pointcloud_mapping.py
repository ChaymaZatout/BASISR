"""
Name : main_pointcloud_mapping.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 20/03/21 03:36 م
Desc: in this example we process the point cloud at once.
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
    # visualize:
    vis = o3d.visualization.Visualizer()
    vis.create_window("BASSAR")
    vis.get_render_option().background_color = [0.75, 0.75, 0.75]
    vis.get_render_option().mesh_show_back_face = True

    # Read data:
    segments = np.load("_in/segments.npy", allow_pickle=True)

    # create bassar:
    small_base = 50
    base = 250
    height = 183.52
    basisr = BASISR(small_base, base, height)

    # add geometries:
    vis.add_geometry(basisr.create_base([255, 255, 255]))
    for p in basisr.pins.flatten():
        if p is not None:
            vis.add_geometry(p)

    # preprocessing:
    # construction the point cloud:
    pcd = np.array(list(sum([s.pcd for s in segments], [])))
    # get the ground height:
    ymin = segments[0].ymin

    # compute heights
    pcd[:, 1] = pcd[:, 1] - ymin

    # processing and mapping pcd:
    start = time.time()
    h = basisr.compute_from_pcd(pcd)
    print(f"Mapping  and computing __time__: {time.time() - start}")

    # updating pins:
    start = time.time()
    basisr.update_pins(h)
    print(f"Updating __time__: {time.time() - start}")

    # show
    vis.run()
    vis.destroy_window()
