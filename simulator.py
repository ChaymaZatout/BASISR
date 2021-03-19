"""
Name : simulator.py
Author : Chayma Zatout
Contact : _
Time    : 09/01/21 08:24 م
Desc:
"""

import numpy as np
import open3d as o3d
from math import cos, sin, pi, atan2, sqrt
import preprocessing


class BASISR:
    colors_dict = {0: [0.85, 0.85, 0.85],
                   1: [0, 1, 0],
                   2: [0, 1, 0],
                   3: [1, 1, 0],
                   4: [1, 0, 0]}

    def __init__(self, small_base, base, height, trX=3, trZ=3, pinRadius=1, pinInitHeight=1, cell=5, diffH=2):
        self.small_base = small_base
        self.base = base
        self.height = height
        self.trX = trX
        self.trZ = trZ
        self.pinRadius = pinRadius
        self.pinInitHeight = pinInitHeight
        self.cell = cell
        self.diffH = diffH

        self.n_lines = int((self.height) / self.trZ)
        self.n_cols = int((self.base) / self.trX)
        self.xLocation = -self.base / 2 + self.pinRadius
        self.zLocation = -self.height + self.pinRadius
        self.pins = BASISR.create_pins(self)
        self.init_y = np.copy(np.asarray(self.pins[int(self.n_lines / 2)][int(self.n_cols / 2)].vertices)[:, 1])

    ####################################################################################################################
    #                                            INITIALIZATION
    ####################################################################################################################
    def create_base(self, rgb_color):
        points = [[-self.small_base / 2, 0, 0], [self.small_base / 2, 0, 0],
                  [self.base / 2, 0, -self.height], [-self.base / 2, 0, -self.height]]
        mesh = [[0, 1, 2], [0, 2, 3]]

        basisr = o3d.geometry.TriangleMesh()
        basisr.vertices = o3d.utility.Vector3dVector(points)
        basisr.triangles = o3d.utility.Vector3iVector(mesh)
        basisr.paint_uniform_color([_ / 255 for _ in rgb_color])
        return basisr

    def create_pins(self):
        h, w = self.n_lines, self.n_cols
        cylinders = np.empty((h, w), object)
        for i in range(w):
            for j in range(h):
                x = self.xLocation + i * self.trX
                z = self.zLocation + j * self.trZ
                if BASISR.is_in(self, x, z):
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=self.pinRadius,
                                                                         height=self.pinInitHeight + self.diffH * 0)
                    cylinder.translate(np.asarray(
                        [BASISR.compute_x(self, i), (self.pinInitHeight + self.diffH * 0) / 2,
                         BASISR.compute_z(self, j)], dtype=float))
                    cylinder.rotate(
                        np.asarray([[1, 0, 0], [0, cos(pi / 2), -sin(pi / 2)], [0, sin(pi / 2), cos(pi / 2)]],
                                   dtype=float))
                    cylinder.paint_uniform_color(BASISR.colors_dict[0])
                    cylinders[j][i] = cylinder
                else:
                    cylinders[j][i] = None
        return cylinders

    def update_pins(self):
        import random
        h, w = self.pins.shape[:2]
        for j in range(h):
            for i in range(w):
                if self.pins[j][i] is not None:
                    cls = random.randint(1, 4)
                    self.update_pin(i, j, cls)

    def update_pin(self, x, y, h):
        nparray = np.asarray(self.pins[y][x].vertices)
        nparray[:, 1] = self.init_y * h * self.diffH
        self.pins[y][x].vertices = o3d.utility.Vector3dVector(nparray)
        self.pins[y][x].paint_uniform_color(BASISR.colors_dict[h])

    @staticmethod
    def compute_x(self, x):
        return self.xLocation + x * self.trX

    @staticmethod
    def compute_z(self, z):
        return self.zLocation + z * self.trZ

    @staticmethod
    def is_in(self, x, z):
        b = self.base / 2 * (-z + (800 * 0.05735)) / (self.height + (800 * 0.05735))
        return -b + self.pinRadius <= x <= b - self.pinRadius

    # source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    @staticmethod
    def rotationMatrixToEulerAngles(R):
        sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = atan2(R[2, 1], R[2, 2])
            y = atan2(-R[2, 0], sy)
            z = atan2(R[1, 0], R[0, 0])
        else:
            x = atan2(-R[1, 2], R[1, 1])
            y = atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    ####################################################################################################################
    #                                            LABELS
    ####################################################################################################################
    def chair(self, centerX, centerZ, height):
        for i in range(centerX - int(self.cell / 2), centerX + int(self.cell / 2) + 1):
            self.pins[centerZ][i].height = height
            self.pins[centerZ][i].is_active = True
            self.pins[centerZ][i].is_centeroid = True

        for i in range(centerZ - int(self.cell / 2), centerZ + int(self.cell / 2) + 1):
            self.pins[i][centerX - int(self.cell / 2)].height = height
            self.pins[i][centerX - int(self.cell / 2)].is_active = True
            self.pins[i][centerX - int(self.cell / 2)].is_centeroid = True

        for i in range(centerZ, centerZ + int(self.cell / 2) + 1):
            self.pins[i][centerX + int(self.cell / 2)].height = height
            self.pins[i][centerX + int(self.cell / 2)].is_active = True
            self.pins[i][centerX + int(self.cell / 2)].is_centeroid = True

    def table(self, centerX, centerZ, height):
        for i in range(centerX - int(self.cell / 2), centerX + int(self.cell / 2) + 1):
            self.pins[centerZ][i].height = height
            self.pins[centerZ][i].is_active = True
            self.pins[centerZ][i].is_centeroid = True

        for i in range(centerZ, centerZ + int(self.cell / 2) + 1):
            self.pins[i][centerX + int(self.cell / 2)].height = height
            self.pins[i][centerX + int(self.cell / 2)].is_active = True
            self.pins[i][centerX + int(self.cell / 2)].is_centeroid = True
            self.pins[i][centerX - int(self.cell / 2)].height = height
            self.pins[i][centerX - int(self.cell / 2)].is_active = True
            self.pins[i][centerX - int(self.cell / 2)].is_centeroid = True

    def dresser(self, centerX, centerZ, height):
        for i in range(self.cell):
            self.pins[centerZ - int(self.cell / 2)][centerX - int(self.cell / 2) + i].height = height  # haut
            self.pins[centerZ - int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_active = True
            self.pins[centerZ - int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_centeroid = True
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].height = height  # bas
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_active = True
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_centeroid = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 2)].height = height  # cote gauche
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 2)].is_active = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 2)].is_centeroid = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 2)].height = height  # cote droit
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 2)].is_active = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 2)].is_centeroid = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX].height = height  # ligne central
            self.pins[centerZ - int(self.cell / 2) + i][centerX].is_active = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX].is_centeroid = True

    def window(self, centerX, centerZ, height):
        for i in range(self.cell):
            self.pins[centerZ - int(self.cell / 2)][centerX - int(self.cell / 2) + i].height = height  # haut
            self.pins[centerZ - int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_active = True
            self.pins[centerZ - int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_centeroid = True
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].height = height  # bas
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_active = True
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_centeroid = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 2)].height = height  # cote gauche
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 2)].is_active = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 2)].is_centeroid = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 2)].height = height  # cote droit
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 2)].is_active = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 2)].is_centeroid = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX].height = height  # ligne central: vetrical
            self.pins[centerZ - int(self.cell / 2) + i][centerX].is_active = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX].is_centeroid = True
            self.pins[centerZ][centerX - int(self.cell / 2) + i].height = height  # bas
            self.pins[centerZ][centerX - int(self.cell / 2) + i].is_active = True
            self.pins[centerZ][centerX - int(self.cell / 2) + i].is_centeroid = True

    def door(self, centerX, centerZ, heigh):
        for i in range(centerX - int(self.cell / 3), centerX + int(self.cell / 3) + 1):
            self.pins[centerZ - int(self.cell / 2)][i].height = heigh
            self.pins[centerZ - int(self.cell / 2)][i].is_active = True
            self.pins[centerZ - int(self.cell / 2)][i].is_centeroid = True
            self.pins[centerZ + int(self.cell / 2)][i].height = heigh
            self.pins[centerZ + int(self.cell / 2)][i].is_active = True
            self.pins[centerZ + int(self.cell / 2)][i].is_centeroid = True

        for i in range(self.cell):
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 3)].height = heigh
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 3)].is_active = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX - int(self.cell / 3)].is_centeroid = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 3)].height = heigh
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 3)].is_active = True
            self.pins[centerZ - int(self.cell / 2) + i][centerX + int(self.cell / 3)].is_centeroid = True

    def upstairs(self, centerX, centerZ, height):
        for i in range(int(self.cell / 2)):
            self.pins[centerZ][centerX + i].height = height
            self.pins[centerZ][centerX + i].is_active = True
            self.pins[centerZ][centerX + i].is_centeroid = True
            self.pins[centerZ + i][centerX].height = height
            self.pins[centerZ + i][centerX].is_active = True
            self.pins[centerZ + i][centerX].is_centeroid = True
            self.pins[centerZ - i][centerX + int(self.cell / 2)].height = height
            self.pins[centerZ - i][centerX + int(self.cell / 2)].is_active = True
            self.pins[centerZ - i][centerX + int(self.cell / 2)].is_centeroid = True
            self.pins[centerZ + int(self.cell / 2)][centerX - i].height = height
            self.pins[centerZ + int(self.cell / 2)][centerX - i].is_active = True
            self.pins[centerZ + int(self.cell / 2)][centerX - i].is_centeroid = True

    def downstairs(self, centerX, centerZ, height):
        for i in range(int(self.cell / 2)):
            self.pins[centerZ][centerX + i].height = height
            self.pins[centerZ][centerX + i].is_active = True
            self.pins[centerZ][centerX + i].is_centeroid = True
            self.pins[centerZ - i][centerX].height = height
            self.pins[centerZ - i][centerX].is_active = True
            self.pins[centerZ - i][centerX].is_centeroid = True
            self.pins[centerZ + i][centerX + int(self.cell / 2)].height = height
            self.pins[centerZ + i][centerX + int(self.cell / 2)].is_active = True
            self.pins[centerZ + i][centerX + int(self.cell / 2)].is_centeroid = True
            self.pins[centerZ - int(self.cell / 2)][centerX - i].height = height
            self.pins[centerZ - int(self.cell / 2)][centerX - i].is_active = True
            self.pins[centerZ - int(self.cell / 2)][centerX - i].is_centeroid = True

    def bathtub5(self, centerX, centerZ, height):
        for i in range(self.cell):
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].height = height
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_active = True
            self.pins[centerZ + int(self.cell / 2)][centerX - int(self.cell / 2) + i].is_centeroid = True

        for i in range(int(self.cell * 2 / 3)):
            self.pins[centerZ + int(self.cell / 2) - i][centerX - int(self.cell / 2)].height = height
            self.pins[centerZ + int(self.cell / 2) - i][centerX - int(self.cell / 2)].is_active = True
            self.pins[centerZ + int(self.cell / 2) - i][centerX - int(self.cell / 2)].is_centeroid = True
            self.pins[centerZ + int(self.cell / 2) - i][centerX + int(self.cell / 2)].height = height
            self.pins[centerZ + int(self.cell / 2) - i][centerX + int(self.cell / 2)].is_active = True
            self.pins[centerZ + int(self.cell / 2) - i][centerX + int(self.cell / 2)].is_centeroid = True

        for i in range(int(self.cell / 3)):
            self.pins[centerZ][centerX - int(self.cell / 6) + i].height = height
            self.pins[centerZ][centerX - int(self.cell / 6) + i].is_active = True
            self.pins[centerZ][centerX - int(self.cell / 6) + i].is_centeroid = True
            self.pins[centerZ + int(self.cell / 2) - int(self.cell * 2 / 3)][
                centerX + int(self.cell / 3) + i].height = height
            self.pins[centerZ + int(self.cell / 2) - int(self.cell * 2 / 3)][
                centerX + int(self.cell / 3) + i].is_active = True
            self.pins[centerZ + int(self.cell / 2) - int(self.cell * 2 / 3)][
                centerX + int(self.cell / 3) + i].is_centeroid = True
            self.pins[centerZ + int(self.cell / 2) - int(self.cell * 2 / 3)][
                centerX - int(self.cell / 3) - i].height = height
            self.pins[centerZ + int(self.cell / 2) - int(self.cell * 2 / 3)][
                centerX - int(self.cell / 3) - i].is_active = True
            self.pins[centerZ + int(self.cell / 2) - int(self.cell * 2 / 3)][
                centerX - int(self.cell / 3) - i].is_centeroid = True

    ####################################################################################################################
    #                                            UPDATE
    ####################################################################################################################

    def update_point(self, x, z, height, color=None):
        i = int(self.n_cols / 2) + int(x / self.trX) if self.n_cols % 2 == 0 else int(
            self.n_cols / 2) + 1 + int(x / self.trX)
        j = self.n_lines + int(z / self.trZ)

        if (0 < i < self.n_cols) and (0 < j < self.n_lines) and self.pins[j][i] != -1 and self.pins[j][
            i].height < self.pinInitHeight + self.diffH:
            self.pins[j][i].height = height
            self.pins[j][i].color = [0, 0, 1] if color is None else color

    def update_pcd(self, pcd, height=None):
        if height is None:
            height = self.pinInitHeight + self.diffH
        for p in pcd:
            self.update_point(p[0], p[2], height)

    def map_segments(self, segments):
        for segment in segments:
            self.map_segment(segment)

    def map_segment(self, segment):
        pcd = preprocessing.map_pcd(segment.pcd)
        heightClass = self.pinInitHeight + self.diffH * (segment.height_class(1200, 1800) + 1)
        if segment.classe is not None:
            self.update_pcd(pcd)
            centeroid = segment.centroid()
            centeroid = preprocessing.map_point(centeroid)
            self.update_centroid(centeroid[0], centeroid[2], heightClass, segment.classe)
        else:
            self.update_pcd(pcd, heightClass)

    def update_centroid(self, x, z, heightClass, objectNature):
        i = int(self.n_cols / 2) + int(x / self.trX) if self.n_cols % 2 == 0 else int(
            self.n_cols / 2) + 1 + int(x / self.trX)
        j = self.n_lines + int(z / self.trZ)
        if (0 < i < self.n_cols) and (0 < j < self.n_lines) and self.pins[j][i] != -1 and self.pins[j][
            i].height < heightClass:
            if objectNature == 'chair':
                self.chair(i, j, heightClass)
            elif objectNature == 'table':
                self.table(i, j, heightClass)
            elif objectNature == 'dresser':
                self.dresser(i, j, heightClass)
            elif objectNature == 'upstair':
                self.upstairs(i, j, heightClass)
            elif objectNature == 'downstair':
                self.downstairs(i, j, heightClass)
            elif objectNature == 'door':
                self.door(i, j, heightClass)
            elif objectNature == 'window':
                self.window(i, j, heightClass)
            elif objectNature == "bathtub5":
                self.bathtub5(i, j, heightClass)


class Pin:
    def __init__(self, xyz, height, is_active=False, is_centeroid=False, color=None):
        self.xyz = xyz
        self.is_active = is_active
        self.height = height
        self.is_centeroid = is_centeroid
        if color is None:
            self.color = [0.85, 0.85, 0.85]


if __name__ == '__main__':
    import time

    # visualize:
    vis = o3d.visualization.Visualizer()
    vis.create_window("BASSAR")
    vis.get_render_option().background_color = [0.75, 0.75, 0.75]
    vis.get_render_option().mesh_show_back_face = True

    # create bassar:
    petiteBase = 50
    grandeBase = 250
    hauteur = 183.52
    start_time = time.time()
    bassar = BASISR(petiteBase, grandeBase, hauteur)
    vis.add_geometry(bassar.create_base([255, 255, 255]))

    # pins:
    pins = bassar.create_pins()
    for p in pins:
        vis.add_geometry(p)
    print('Creation and initialisation of BASISR: ' + str(time.time() - start_time) + ' seconds')

    # show
    vis.run()
    vis.destroy_window()
