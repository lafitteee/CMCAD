from ast import arg
import ezdxf
from ezdxf.addons import r12writer
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def generate_one(cmd, arg, id, save_dir='img2dxf_GT'):
    save_name = save_dir + str(id) + ".dxf"
    save_dir = os.path.join('F:\Lafitte\Projects\MyDeepCAD\generated_dxf', save_dir)
    with r12writer(os.path.join(save_dir, save_name)) as dxf:
        for i in range(len(cmd)):
            if cmd[i] == 1:
                x1 = arg[i][0]
                y1 = arg[i][1]
                x2 = arg[i][2]
                y2 = arg[i][3]
                dxf.add_line((x1, y1), (x2, y2))
            elif cmd[i] == 2:
                x = arg[i][0]
                y = arg[i][1]
                start = arg[i][4] / 255 * 360
                end = arg[i][5] / 255 * 360
                radius = arg[i][6]
                dxf.add_arc((x, y), radius=radius, start=start, end=end)
            elif cmd[i] == 3:
                x = arg[i][0]
                y = arg[i][1]
                radius = arg[i][6]
                dxf.add_circle((x, y), radius=radius)


def generate_dxf(cmds, args):
    for j in tqdm(range(len(cmds))):
        cmd = cmds[j]
        arg = args[j]
        save_name = "img2dxf_GT" + str(j) + ".dxf"
        with r12writer(os.path.join( "F:\Lafitte\Projects\MyDeepCAD\generated_dxf\img2dxf_GT", save_name)) as dxf:
            for i in range(len(cmd)):
                if cmd[i] == 1:
                    x1 = arg[i][0]
                    y1 = arg[i][1]
                    x2 = arg[i][2]
                    y2 = arg[i][3]
                    dxf.add_line((x1, y1), (x2, y2))
                elif cmd[i] == 2:
                    x = arg[i][0]
                    y = arg[i][1]
                    start = arg[i][4] / 255 * 360
                    end = arg[i][5] / 255 * 360
                    radius = arg[i][6]
                    dxf.add_arc((x, y), radius=radius, start=start, end=end)
                elif cmd[i] == 3:
                    x = arg[i][0]
                    y = arg[i][1]
                    radius = arg[i][6]
                    dxf.add_circle((x, y), radius=radius)


if __name__ == '__main__':
    cmds = np.load('./output/img2dxf_cmds_GroundTruth_2023-02-08-17-33-48.npy', allow_pickle=True)
    args = np.load('./output/img2dxf_args_GroundTruth_2023-02-08-17-33-48.npy', allow_pickle=True)
    generate_dxf(cmds, args)
