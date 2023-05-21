import numpy as np
import pandas as pd
import json
import math
from tqdm import tqdm


def process_LINE(entity) -> list:
    ret = [None] * 7
    ret[0] = round(entity.dxf.start[0], 2)
    ret[1] = round(entity.dxf.start[1], 2)
    ret[2] = round(entity.dxf.end[0], 2)
    ret[3] = round(entity.dxf.end[1], 2)
    return ret


def process_ARC(entity) -> list:
    ret = [None] * 7
    # ret[0] = round(entity.start_point[0], 2)
    # ret[1] = round(entity.start_point[1], 2)
    # ret[2] = round(entity.end_point[0], 2)
    # ret[3] = round(entity.end_point[1], 2)
    ret[0] = round(entity.dxf.center[0], 2)
    ret[1] = round(entity.dxf.center[1], 2)
    ret[4] = round(entity.dxf.start_angle, 1)
    ret[5] = round(entity.dxf.end_angle, 1)
    ret[6] = round(entity.dxf.radius, 2)
    return ret


def process_CIRCLE(entity) -> list:
    ret = [None] * 7
    ret[0] = round(entity.dxf.center[0], 2)
    ret[1] = round(entity.dxf.center[1], 2)
    ret[6] = round(entity.dxf.radius, 2)
    return ret


def recur_explode_polyline(polylines):
    if len(polylines) > 0:
        for pl in polylines:
            subset = pl.explode()  # explode()返回一个query container
            recur_explode_polyline(subset.query('LWPOLYLINE' or 'POLYLINE'))  # query container也能进行query


def recur_explode_INSERT(blocks):
    if len(blocks) > 0:
        for block in blocks:
            subset = block.explode()
            recur_explode_INSERT(subset.query('INSERT'))


def process_single_dxf(msp):
    # 先把块炸开
    blocks = msp.query('INSERT')
    try:
        recur_explode_INSERT(blocks)
    except Exception as e:
        # print(e)
        pass
    # 先把所有的多段线递归炸开
    pls = msp.query('LWPOLYLINE' or 'POLYLINE')
    recur_explode_polyline(pls)

    # 初始值为<Start>标志
    params = [[None]*7]
    command = [0]
    for e in msp:
        if e.dxftype() == 'LINE':
            command.append(1)
            params.append(process_LINE(e))
        elif e.dxftype() == 'ARC':
            command.append(2)
            params.append(process_ARC(e))
        elif e.dxftype() == 'CIRCLE':
            command.append(3)
            params.append(process_CIRCLE(e))
    # print(np.array(command).shape)
    # print(np.array(params).shape)
    while len(command) < 62:
        command.append(4)
        params.append([None]*7)
    return command, params


def normalizeSingleDxf(paths, scale_target=255, decimals=0):
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for path in paths:  # 记录所有的坐标中的最值
        if path[0] != None:
            min_x = min(min_x, path[0])
            min_y = min(min_y, path[1])
            max_x = max(max_x, path[0])
            max_y = max(max_y, path[1])
        if path[2] != None:
            min_x = min(min_x, path[2])
            min_y = min(min_y, path[3])
            max_x = max(max_x, path[2])
            max_y = max(max_y, path[3])
    # 平移缩放
    max_x -= min_x
    max_y -= min_y
    try:
        scale = float(scale_target / max(max_x, max_y))  # scale_target是缩放后坐标的最大值
    except:
        return 1  # 丢弃这个图形
    # print(scale)
    for path in paths:
        # 平移
        if path[0] is not None:
            path[0] -= min_x
            path[1] -= min_y
        if path[2] is not None:
            path[2] -= min_x
            path[3] -= min_y
        # 缩放
        if path[0] is not None:
            path[0] = path[0] * scale
            path[1] = path[1] * scale
        if path[2] is not None:
            path[2] = path[2] * scale
            path[3] = path[3] * scale
        if path[6] is not None:
            path[6] = path[6] * scale
        # 把角度的负值变为正数，不会影响图形，然后把角度也缩放到[0, 255]
        if path[4] is not None:
            if path[4] < 0:
                path[4] = path[4] + 360.0
            path[4] = path[4] / 360.0 * 255
        if path[5] is not None:
            if path[5] < 0:
                path[5] = path[5] + 360.0
            path[5] = path[5] / 360.0 * 255
            # print(path[0], path[1], path[2], path[3], path[4], path[5], path[6])
        for i in range(7):
            if path[i] is not None:
                path[i] = round(path[i], decimals)
    
    return 0

