"""
Name : non_blocking_viz.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 12/03/21 01:27 م
Desc:
"""
import open3d as o3d
from simulator import BASISR
import numpy as np

if __name__ == '__main__':
    # Create Open3d visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.create_window("BASSAR")
    vis.get_render_option().background_color = [0.75, 0.75, 0.75]
    vis.get_render_option().mesh_show_back_face = True

    # create bassar:
    small_base = 50
    base = 250
    height = 183.52
    bassar = BASISR(small_base, base, height)
    vis.add_geometry(bassar.create_base([255, 255, 255]))

    # add pins:
    for j in range(bassar.pins.shape[0]):
        for i in range(bassar.pins.shape[1]):
            if bassar.pins[j][i] is not None:
                vis.add_geometry(bassar.pins[j][i])

    while True:
        heights = np.random.randint(0, 4, [bassar.n_lines, bassar.n_cols])
        bassar.update_pins(heights)
        for p in bassar.pins.flatten():
            vis.update_geometry(p)
        vis.poll_events()
        vis.update_renderer()
