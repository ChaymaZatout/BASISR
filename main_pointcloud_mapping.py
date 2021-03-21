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
    # to compute a point class:
    bf = 1200 * 0.0575
    bs = 1800 * 0.0575

    # add geometries:
    vis.add_geometry(basisr.create_base([255, 255, 255]))
    for p in basisr.pins.flatten():
        if p is not None:
            vis.add_geometry(p)

    # preprocessing:
    # construction the point cloud:
    pcd = np.array(list(sum([s.pcd for s in segments], [])))
    # get the ground height:
    ymin = segments[2].ymin

    # processing and mapping pcd:
    start = time.time()
    pcd[:, 1] = pcd[:, 1] - ymin  # compute heights
    pcd = map_pcd(pcd)  # map pcd.
    j, i = get_indices(pcd, basisr.n_lines, basisr.n_cols, basisr.trX, basisr.trZ)
    # compute classes:
    h = np.zeros((basisr.n_lines, basisr.n_cols))
    h[j, i] = np.where(pcd[:, 1] < bf, 1,
                       np.where((pcd[:, 1] >= bf) & (pcd[:, 1] < bs), 2, np.where(pcd[:, 1] > bs, 3, 0)))
    print(f"Mapping  and computing __time__: {time.time() - start}")

    # updating pins:
    start = time.time()
    basisr.update_pins(h)
    print(f"Updating __time__: {time.time() - start}")

    # show
    vis.run()
    vis.destroy_window()
