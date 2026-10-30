# -*- coding: utf-8 -*-
"""
红线自动匹配工具
================
从"无人机正射影像图"(已带红线)中提取红线多边形, 利用无人机照片的
RTK 定位与云台姿态信息 + 图像特征匹配, 把红线精确转绘到同编号的
无人机特写照片上, 批量输出。

原理概述
--------
1. 红线提取: 纯红色阈值分割 + 图角图例框剔除(图例红样例不参与匹配) +
   孤立红色小标记剔除 + 碎片就近归组(掩膜文字打断的红线属同一个圈) +
   递增内核闭运算跨断口封口 + 孔洞填充 + 多边形化,
   支持一个地块多条红线; 自动排除红屋顶等暗红色干扰(纯度检验)。
2. 配准: 将照片按云台航向角预旋转至"北向上", 用 RootSIFT 与正射图
   红线周边区域做特征匹配, 估计相似变换; 多照片尺度一致性 + 相机模型
   双重门控剔除误匹配; 用通过门控的匹配建立 正射图像素 <-> 地面米制
   坐标 的配准关系。
3. 转绘: 对每张照片, 用针孔相机模型(焦距/云台姿态/RTK位置/地面高程)
   把红线逐点投影到照片; 再沿红线做模板匹配(NCC)局部精化平移。
4. 输出: 按编号分文件夹保存带红线照片(保留EXIF), 并生成对照图与
   处理报告 CSV。

用法
----
图形界面:   双击运行  redline_matcher.exe  (或 python redline_matcher.py)
命令行:     redline_matcher.exe --cli --ref 正射文件夹 --photo 照片文件夹 --out 输出文件夹
"""
import argparse
import csv
import math
import os
import re
import shutil
import sys
import threading
import time
import traceback

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)

# ----------------------------- 常量 -----------------------------
IMG_EXTS = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')
MATCH_SIZE = 2200          # 特征匹配时的图像最长边
MIN_INLIERS = 30           # 锚点匹配最少内点
MIN_INLIERS_SINGLE = 60    # 仅有一个候选时的最少内点
SCALE_TOL = 0.15           # 尺度一致性容差
MODEL_GATE_M = 6.0         # 配准后模型门控(米)
NCC_PATCH = 160            # NCC 精化补丁边长(正射像素)
NCC_SEARCH_M = 6.0         # NCC 搜索半径(米)
NCC_MIN_PEAK = 0.45        # NCC 峰值阈值
NCC_MIN_PATCHES = 3        # 精化最少有效补丁数
NCC_MAX_SHIFT_M = 12.0     # 精化允许的最大位移(米)
FOCAL_MM = 12.29           # DJI M3E 广角相机焦距
SENSOR_W_MM = 17.3         # 4/3 传感器宽
M_PER_LAT = 111320.0


def _resource_path(relative_path):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


# ----------------------------- 基础IO -----------------------------
def imread_cn(path):
    """支持中文路径的图像读取"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_cn(path, img, quality=92):
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMG_EXTS:
        ext = '.jpg'
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(path)
    return ok


def _digits_not_year(name):
    """返回名称中的数字串列表, 排除年份(如 2025年 / 2026年 / 2025 年)"""
    out = []
    for m in re.finditer(r'\d+', name):
        if re.match(r'\s*年', name[m.end():]):
            continue  # 数字后面紧跟"年", 视为年份, 不作为编号
        out.append(m.group(0))
    return out


def get_plot_id(name):
    """从文件名或文件夹名提取地块编号: 优先 XXXXX号(图斑) 中的数字, 位数不限;
    年份(如2025年)不作为编号"""
    m = re.search(r'(\d+)\s*号', name)
    if m:
        return m.group(1)
    cands = _digits_not_year(name)
    return cands[0] if cands else None


# ----------------------------- 红线提取 -----------------------------
def _fill_holes(mask):
    h, w = mask.shape
    ff = mask.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 128)
    holes = (ff != 128).astype(np.uint8) * 255
    return cv2.bitwise_or(mask, holes)


def erase_annotation(img, log=print):
    """抹除正射图上的经纬度标注文字和绿色中心点(修复其对红线的遮挡断口)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gmask = cv2.inRange(hsv, (40, 120, 120), (90, 255, 255))
    cnts, _ = cv2.findContours(gmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        a = cv2.contourArea(c)
        if not (25 <= a <= 6000):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if 0.35 < w / max(h, 1) < 2.8:      # 近圆形
            dot = (x, y, w, h)
            break
    if dot is None:
        return img
    x, y, w, h = dot
    W, H = img.shape[1], img.shape[0]
    # 标注文字一般在绿点邻近位置, 开一个局部窗口处理, 避免误伤指北针等
    x0 = max(0, x - 250); y0 = max(0, y - 260)
    x1 = min(W, x + 950); y1 = min(H, y + 320)
    roi = img[y0:y1, x0:x1]
    r, g, b = roi[:, :, 2], roi[:, :, 1], roi[:, :, 0]
    white = ((r > 185) & (g > 185) & (b > 185)).astype(np.uint8) * 255
    green = gmask[y0:y1, x0:x1]
    mask = cv2.dilate(white | green, np.ones((9, 9), np.uint8))
    # 红色像素(可能是紧邻标记点的红线)不抹除
    redish = ((r > 150) & ((r - np.maximum(g, b).astype(np.int16)) > 80))
    mask[redish] = 0
    if mask.sum() < 500:
        return img
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.inpaint(roi, mask, 6, cv2.INPAINT_TELEA)
    log(f'  已抹除经纬度标注与中心点标记(窗口 {x1 - x0}x{y1 - y0})')
    return out


def detect_legend_boxes(img, log=print):
    """检测图角处的图例白底矩形框(右下角/左下角等), 返回 [(x,y,w,h)].
    特征: 紧贴某个图角、矩形度高的大块亮区(白底), 内部含黑字/黑框.
    图例中的红色样例不是地块红线, 不参与匹配(目标红线在画面中部)."""
    H, W = img.shape[:2]
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    bright = ((r > 170) & (g > 170) & (b > 170)).astype(np.uint8) * 255
    # 只做小开运算去噪; 不可用大内核闭运算(亮区会连成一片淹没图例框)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if not (0.0012 * H * W <= a <= 0.08 * H * W):   # 图例是紧凑小框, 非整幅亮区
            continue
        if w > 0.5 * W or h > 0.5 * H or a < 0.75 * w * h:
            continue
        dc = min(math.hypot(x, y), math.hypot(W - x - w, y),
                 math.hypot(x, H - y - h), math.hypot(W - x - w, H - y - h))
        if dc > 0.12 * min(H, W):                # 必须紧贴某个图角
            continue
        roi = img[y:y + h, x:x + w]
        if float((roi.mean(axis=2) < 90).mean()) < 0.003:   # 图例必有黑字/黑框
            continue
        boxes.append((x, y, w, h))
    return boxes


def extract_redlines(img, log=print):
    """提取红线多边形列表 [(poly Nx1x2 int32, area, purity)], 按面积降序.
    流程: 抹除经纬度标注 -> 剔除图角图例框内红色 -> 红色阈值分割 ->
    丢弃孤立红色小标记 -> 碎片就近归组(被掩膜文字打断的红线属同一个圈) ->
    每组递增内核闭运算跨断口封口填充成环 / RANSAC 直线簇拟合(允许任意大小断口)."""
    img = erase_annotation(img, log)
    b, g, r = (img[:, :, i].astype(np.int16) for i in range(3))
    mx = np.maximum(g, b)
    purity = r - mx
    mask = ((r > 150) & (purity > 90) & (mx < 130)).astype(np.uint8) * 255
    H, W = mask.shape
    # 图例框内的红色样例直接抹除, 不参与红线匹配
    for (bx, by, lbw, lbh) in detect_legend_boxes(img, log):
        pad = max(12, int(0.012 * min(H, W)))
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(W, bx + lbw + pad), min(H, by + lbh + pad)
        if int(mask[y0:y1, x0:x1].sum()) > 0:
            log(f'  忽略角落图例框内的红色样例 (框 {lbw}x{lbh} @({bx},{by}))')
            mask[y0:y1, x0:x1] = 0
    # 轻度闭运算连接邻近线段(包括被标记点打断的小红线), 然后取连通域
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n <= 1:
        return []
    comps = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 300:                            # 绝对下限即可, 不用相对比例(红线可以很小)
            continue
        sel = (labels == i) & (mask > 0)       # 原始红色像素(不含闭运算填充)
        if sel.sum() < 40:
            continue
        mp = float(np.mean(purity[sel]))
        if mp < 175:                           # 排除屋顶/红土等暗红区域(标注红线纯度≥200)
            continue
        ys, xs = np.nonzero(sel)
        comps.append(dict(sel=sel, pts=np.stack([xs, ys], axis=1).astype(np.float64),
                          area=float(a), purity=mp, raw=int(sel.sum()),
                          bbox=(int(stats[i, 0]), int(stats[i, 1]),
                                int(stats[i, 2]), int(stats[i, 3]))))
    # 丢弃孤立红色小标记(小红旗/箭头等): 自身很小且紧邻一个明显更大的红线体
    keep = []
    for c in comps:
        cx, cy, cw, ch = c['bbox']
        is_marker = False
        if c['raw'] < 800 and max(cw, ch) < 80:
            for d in comps:
                if d is c or d['raw'] < 8 * c['raw']:
                    continue
                dx, dy, dw, dh = d['bbox']
                gap = max(0, max(cx - dx - dw, dx - cx - cw,
                                 cy - dy - dh, dy - cy - ch))
                if gap < 120:
                    is_marker = True
                    break
        if is_marker:
            log(f'  忽略红色小标记 ({cw}x{ch} @({cx},{cy}), 近旁有主红线)')
        else:
            keep.append(c)
    comps = keep

    polys = []
    frags = []
    for c in comps:
        if max(c['bbox'][2], c['bbox'][3]) < 120:   # 点状标记: 保留原始轮廓
            poly = _raw_contour_poly(c['sel'])
            if poly is not None and len(poly) >= 3:
                polys.append((poly, c['area'], c['purity']))
        else:
            frags.append(c)
    # 碎片就近归组: 掩膜文字会把同一条红线(一个圈)打断成多段, 段间距离近者归为一组
    if frags:
        bridge = max(120, int(0.06 * min(H, W)))
        fragmask = np.zeros((H, W), np.uint8)
        for c in frags:
            fragmask[c['sel']] = 1
        dil = cv2.dilate(fragmask, np.ones((2 * bridge + 1, 2 * bridge + 1), np.uint8))
        _, glab = cv2.connectedComponents(dil, 8)
        groups = {}
        for c in frags:
            g = int(np.bincount(glab[c['sel']].ravel()).argmax())
            groups.setdefault(g, []).append(c)
        for g, members in sorted(groups.items()):
            sel_u = np.zeros((H, W), bool)
            for c in members:
                sel_u |= c['sel']
            pts_u = np.vstack([c['pts'] for c in members])
            area_u = sum(c['area'] for c in members)
            mp_u = float(np.mean([c['purity'] for c in members]))
            if len(members) > 1:
                log(f'  红线碎片归组: {len(members)} 段合并封闭(跨越文字掩膜断口)')
            got = fill_simplify_polygons(sel_u, log)
            if not got:
                # 断口环: RANSAC 直线簇拟合(允许任意大小断口)
                poly = polygonize_ransac(pts_u, log, sel_u)
                got = [poly] if poly is not None else []
            if not got:
                big = max(members, key=lambda c: c['raw'])
                poly = _raw_contour_poly(big['sel'])
                got = [poly] if poly is not None else []
            for poly in got:
                if poly is not None and len(poly) >= 3:
                    polys.append((poly, area_u, mp_u))
    polys.sort(key=lambda t: -t[1])
    return polys


def simplify_polygon(c, tol_px=12.0):
    """对单线闭合轮廓做自适应简化: 在顶点数稳定平台中选取,
    保真度(轮廓点到多边形边的95分位距离)不超过 tol_px 的最简者"""
    peri = cv2.arcLength(c, True)
    if peri <= 0:
        return None
    pts = c.reshape(-1, 2)
    if len(pts) > 3000:
        pts = pts[::len(pts) // 3000 + 1]

    def fidelity(poly):
        d = [abs(cv2.pointPolygonTest(poly.astype(np.float32),
                                      (float(p[0]), float(p[1])), True))
             for p in pts[::max(1, len(pts) // 1500)]]
        return float(np.percentile(d, 95)) if d else 0.0

    seq = []
    for i in range(48):
        eps = peri * (0.0005 * (1.13 ** i))
        p = cv2.approxPolyDP(c, eps, True)
        seq.append((len(p), p))
    # 提取平台(顶点数连续 >=3 次不变)
    plats = []
    cur = [seq[0]]
    for n, p in seq[1:]:
        if n == cur[-1][0]:
            cur.append((n, p))
        else:
            plats.append(cur)
            cur = [(n, p)]
    plats.append(cur)
    cands = [pl[len(pl) // 2] for pl in plats if len(pl) >= 3 and pl[0][0] >= 3]
    if not cands:
        return cv2.approxPolyDP(c, 0.002 * peri, True).astype(np.int32)
    # 顶点数从多到少(保真优先), 第一个满足保真度阈值的即采用
    for n, p in sorted(cands, key=lambda t: -t[0]):
        if fidelity(p) <= tol_px:
            return p.astype(np.int32)
    return cands[0][1].astype(np.int32)


def fill_simplify_polygons(sel, log):
    """红线环若能封闭: 局部闭运算(内核递增, 允许跨文字掩膜大断口)+填充 ->
    单线外轮廓 -> 自适应简化. 封闭判据与形状无关: 填充后轮廓面积 远大于 笔画面积.
    返回多边形列表(一组归组红线可能含多个独立闭合环), 无法封闭返回 []"""
    stroke_px = int(sel.sum())
    if stroke_px < 50:
        return []
    ys, xs = np.nonzero(sel)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    H, W = sel.shape
    for k in (13, 25, 41, 61, 91, 121, 161, 211, 281):
        pad = max(260, k + 30)
        xa = max(0, x0 - pad); ya = max(0, y0 - pad)
        xb = min(W, x1 + pad); yb = min(H, y1 + pad)
        m = sel[ya:yb, xa:xb].astype(np.uint8) * 255
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        filled = _fill_holes(closed)
        cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 2000]
        if not cnts:
            continue
        # 封闭成功: 填充面积显著大于笔画面积(细长/L形多边形也成立)
        good = [c for c in cnts if cv2.contourArea(c) > 3.0 * stroke_px]
        if not good:
            continue
        out = []
        for cbig in sorted(good, key=cv2.contourArea, reverse=True):
            poly = simplify_polygon(cbig)
            if poly is not None:
                poly = poly.reshape(-1, 1, 2) + np.array([[[xa, ya]]], np.int32)
                out.append(poly.astype(np.int32))
        if out:
            log(f'  红线封闭环提取: {len(out)} 个环, 顶点 {[len(p) for p in out]} (闭运算核{k})')
            return out
    return []


def _raw_contour_poly(sel_mask):
    cnts, _ = cv2.findContours(sel_mask.astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    return cv2.approxPolyDP(c, max(1.5, 0.004 * peri), True).astype(np.int32)


def _contour_guided_order(segs, sel_mask):
    """用原始轮廓的行走顺序给边排序(轮廓沿边界走, 顺序即边界序)"""
    cnts, _ = cv2.findContours(sel_mask.astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c) < 100:
        return None
    c = c[::max(1, len(c) // 4000)]
    assign = np.full(len(c), -1, dtype=int)
    dmin = np.full(len(c), 1e18)
    for si, s in enumerate(segs):
        a, b = s[0], s[1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            continue
        t = np.clip(((c - a) @ ab) / L2, 0, 1)
        dist = np.linalg.norm(c - (a + t[:, None] * ab), axis=1)
        upd = dist < dmin
        dmin[upd] = dist[upd]
        assign[upd] = si
    assign[dmin > 20] = -1
    # 去毛刺: 与前后都不同的孤立点归并到前一个
    for i in range(1, len(assign) - 1):
        if assign[i] != assign[i - 1] and assign[i] != assign[i + 1]:
            assign[i] = assign[i - 1]
    order = []
    seen = set()
    cnt_id = {}
    for a in assign:
        if a < 0:
            continue
        cnt_id[a] = cnt_id.get(a, 0) + 1
        if a not in seen:
            seen.add(a)
            order.append(a)
    total = max(1, sum(cnt_id.values()))
    order = [a for a in order if cnt_id[a] >= 0.02 * total]
    return order if len(order) >= 3 else None


def _greedy_chain_order(segs):
    """端点贪心链接得到边序(备用)"""
    ids = sorted(range(len(segs)), key=lambda i: -np.linalg.norm(segs[i][1] - segs[i][0]))
    chain = [ids[0]]
    remaining = ids[1:]
    while remaining:
        tail = segs[chain[-1]][1]
        nxt = min(range(len(remaining)),
                  key=lambda k: min(np.linalg.norm(segs[remaining[k]][0] - tail),
                                    np.linalg.norm(segs[remaining[k]][1] - tail)))
        chain.append(remaining.pop(nxt))
    return chain


def polygonize_ransac(pts, log=lambda s: None, sel_mask=None, max_lines=30):
    """RANSAC 逐边拟合直线簇 -> 轮廓引导/端点链接排序 -> 相邻边求交 -> 规则矢量多边形.
    边数自适应(弯曲边界保留更多段); 失败返回 None"""
    P = pts
    if len(P) > 30000:
        idx = np.random.default_rng(7).choice(len(P), 30000, replace=False)
        P = P[idx]
    rest = P.copy()
    lines = []            # (point, dir_unit, n_inliers, cluster_pts)
    band = 9.0
    min_inl = max(45, int(0.004 * len(P)))
    rng = np.random.default_rng(42)
    for _ in range(max_lines):
        if len(rest) < min_inl * 2:
            break
        best = None
        m = len(rest)
        for _ in range(1600):
            i, j = rng.integers(0, m, 2)
            if i == j:
                continue
            p1, p2 = rest[i], rest[j]
            d = p2 - p1
            L = np.hypot(d[0], d[1])
            if L < 60:                     # 采样点对太近无意义
                continue
            d /= L
            dist = np.abs((rest[:, 0] - p1[0]) * d[1] - (rest[:, 1] - p1[1]) * d[0])
            inl = dist < band
            cnt = int(inl.sum())
            if best is None or cnt > best[0]:
                best = (cnt, inl, p1, d)
        if best is None or best[0] < min_inl:
            break
        cnt, inl, p1, d = best
        lines.append((p1, d, cnt, rest[inl]))
        rest = rest[~inl]
    if len(lines) < 3:
        return None
    # 去重: 近平行(6°)且间距<25 的线视为同一边, 留支持度大的(与顺序无关)
    keep = []
    for it in sorted(lines, key=lambda t: -t[2]):
        p1, d, cnt, cp = it
        dup = False
        for k in keep:
            if abs(float(d @ k[1])) > math.cos(math.radians(6)):
                sep = abs((p1[0] - k[0][0]) * k[1][1] - (p1[1] - k[0][1]) * k[1][0])
                if sep < 25:
                    dup = True
                    break
        if not dup:
            keep.append(it)
    # 每条线 -> 线段(点簇投影范围)
    segs = []
    for p1, d, cnt, cp in keep:
        pr = (cp - p1) @ d
        lo, hi = pr.min(), pr.max()
        if hi - lo < 30:
            continue
        segs.append([p1 + lo * d, p1 + hi * d, d, cnt])
    if len(segs) < 3:
        return None

    def near_seg(cand, s):
        dvec = s[1] - s[0]
        L = np.linalg.norm(dvec)
        if L < 1e-6:
            return False
        du = dvec / L
        pr = float((cand - s[0]) @ du)
        lat = abs(float((cand - s[0]) @ np.array([-du[1], du[0]])))
        return -0.5 * L <= pr <= 1.5 * L and lat < max(80.0, 0.4 * L)

    def build_poly(order):
        """按给定边序装配多边形: 相邻边直线求交(须在两边附近), 否则取端点中点"""
        verts = []
        n = len(order)
        for i in range(n):
            sA = segs[order[i]]
            sB = segs[order[(i + 1) % n]]
            pA, dA = sA[0], sA[2]
            pB, dB = sB[0], sB[2]
            Amat = np.array([[dA[0], -dB[0]], [dA[1], -dB[1]]])
            det = np.linalg.det(Amat)
            v = None
            if abs(det) > math.sin(math.radians(5)):
                t = np.linalg.solve(Amat, pB - pA)
                cand = pA + t[0] * dA
                if near_seg(cand, sA) and near_seg(cand, sB):
                    v = cand
            if v is None:
                # 两线段最近端点对的中点(含跨断口连接)
                cands_ = [(sA[1], sB[0]), (sA[1], sB[1]), (sA[0], sB[0]), (sA[0], sB[1])]
                a2, b2 = min(cands_, key=lambda t: np.linalg.norm(t[0] - t[1]))
                v = (a2 + b2) / 2
            verts.append(v)
        poly = np.array(verts)
        if _poly_self_intersects(poly):
            return None
        return poly

    # 边排序策略1: 原始轮廓引导(沿真实边界走, 任意形状都不会错序, 杜绝双线/交叉)
    poly = None
    if sel_mask is not None:
        order = _contour_guided_order(segs, sel_mask)
        if order is not None and len(order) >= 3:
            poly = build_poly(order)
    # 策略2: 端点贪心链接序
    if poly is None:
        order = _greedy_chain_order(segs)
        poly = build_poly(order)
    if poly is None:
        return None
    # 覆盖度评估: 多边形边采样点落在红线附近的比例
    samples = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        L = np.linalg.norm(b - a)
        for t in np.linspace(0, 1, max(2, int(L / 15)), endpoint=False):
            samples.append(a * (1 - t) + b * t)
    if samples:
        S = np.array(samples)
        # 每个采样到最近红点的距离(分块算, 防内存过大)
        ok = 0
        step = max(1, len(P) // 8000)
        Q = P[::step]
        for s0 in range(0, len(S), 500):
            chunk = S[s0:s0 + 500]
            D = np.linalg.norm(chunk[:, None, :] - Q[None, :, :], axis=2).min(axis=1)
            ok += int((D < 20).sum())
        log(f'  多边形化: {len(poly)} 顶点, 边覆盖度 {ok / len(S):.0%}')
    return poly.reshape(-1, 1, 2).astype(np.int32)


def _seg_intersect(a1, a2, b1, b2):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _poly_self_intersects(poly):
    n = len(poly)
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j in (i, (i + 1) % n) or (j + 1) % n == i:
                continue
            b1, b2 = poly[j], poly[(j + 1) % n]
            if _seg_intersect(a1, a2, b1, b2):
                return True
    return False


# ----------------------------- 照片元数据 -----------------------------
def read_photo_meta(path):
    """读取 DJI XMP 云台/定位信息 + EXIF 焦距/变焦; 返回 dict 或 None(非无人机照片)"""
    try:
        with open(path, 'rb') as f:
            s = f.read(400000).decode('latin1', errors='ignore')
        meta = {}
        for k in ('GimbalYawDegree', 'GimbalPitchDegree', 'GimbalRollDegree',
                  'RelativeAltitude', 'AbsoluteAltitude', 'GpsLatitude', 'GpsLongitude'):
            m = re.search(r'drone-dji:' + k + r'="([^"]+)"', s) or re.search(k + r'="([^"]+)"', s)
            meta[k] = float(m.group(1)) if m else None
        if meta['GpsLatitude'] is None or meta['GimbalYawDegree'] is None:
            return None
    except Exception:
        return None
    # EXIF 焦距与数码变焦(御3E 有广角/长焦两个相机, 必须逐张读取)
    meta['FocalLength'] = None
    meta['FocalLengthIn35mm'] = None
    meta['DigitalZoomRatio'] = 1.0
    try:
        from PIL import Image
        ifd = Image.open(path).getexif().get_ifd(0x8769)
        if ifd.get(0x920A):
            meta['FocalLength'] = float(ifd[0x920A])
        if ifd.get(0xA405):
            meta['FocalLengthIn35mm'] = float(ifd[0xA405])
        if ifd.get(0xA404):
            meta['DigitalZoomRatio'] = float(ifd[0xA404]) or 1.0
    except Exception:
        pass
    return meta


def eff_f_px(meta, w):
    """照片有效焦距(像素): 覆盖广角/长焦相机与数码变焦"""
    zoom = meta.get('DigitalZoomRatio') or 1.0
    flmm = meta.get('FocalLength')
    if flmm and abs(flmm - 12.29) < 0.1 and w > 5000:
        return 12.29 / 17.3 * w * zoom          # 御3E 广角主摄(4/3", 精确常数)
    f35 = meta.get('FocalLengthIn35mm') or 24.0
    return f35 / 36.0 * w * zoom                # 其余(长焦等)按35mm等效换算


# ----------------------------- 相机模型 -----------------------------
def camera_frames(yaw_deg, pitch_deg):
    """DJI 云台角 -> ENU 坐标系下 (图像右r, 图像下v, 光轴d) 单位向量"""
    psi = math.radians(yaw_deg)
    tau = math.radians(90.0 + pitch_deg)     # 离天底角
    d = np.array([math.sin(psi) * math.sin(tau), math.cos(psi) * math.sin(tau), -math.cos(tau)])
    r = np.array([math.cos(psi), -math.sin(psi), 0.0])
    v = np.cross(d, r)
    return r, v, d


def photo_px_to_ground(u, v, meta, w, h, elev, lat0, lon0):
    """照片像素 -> 地面 ENU 米坐标 (z=elev 平面); 视线太平缓时返回 None"""
    f = eff_f_px(meta, w)
    r, vv, d = camera_frames(meta['GimbalYawDegree'], meta['GimbalPitchDegree'])
    R_wc = np.column_stack([r, vv, d])
    ray = R_wc @ np.array([(u - w / 2) / f, (v - h / 2) / f, 1.0])
    ray /= np.linalg.norm(ray)
    if ray[2] > -0.35:           # 视线与水平面夹角 < ~20°, 放弃
        return None
    m_per_lon = M_PER_LAT * math.cos(math.radians(lat0))
    C = np.array([(meta['GpsLongitude'] - lon0) * m_per_lon,
                  (meta['GpsLatitude'] - lat0) * M_PER_LAT,
                  meta['AbsoluteAltitude']])
    t = (elev - C[2]) / ray[2]
    if t <= 0:
        return None
    P = C + t * ray
    return float(P[0]), float(P[1])


def ground_to_photo_px(E, N, meta, w, h, elev, lat0, lon0):
    """地面 ENU 米坐标 -> 照片像素; 在相机背后返回 None"""
    f = eff_f_px(meta, w)
    r, vv, d = camera_frames(meta['GimbalYawDegree'], meta['GimbalPitchDegree'])
    R_cw = np.vstack([r, vv, d])
    m_per_lon = M_PER_LAT * math.cos(math.radians(lat0))
    C = np.array([(meta['GpsLongitude'] - lon0) * m_per_lon,
                  (meta['GpsLatitude'] - lat0) * M_PER_LAT,
                  meta['AbsoluteAltitude']])
    c = R_cw @ (np.array([E, N, elev]) - C)
    if c[2] <= 1e-6:
        return None
    return float(f * c[0] / c[2] + w / 2), float(f * c[1] / c[2] + h / 2)


def axis_ground_point(meta, w, h, elev, lat0, lon0):
    """相机光轴与地面的交点 (ENU)"""
    return photo_px_to_ground(w / 2, h / 2, meta, w, h, elev, lat0, lon0)


# ----------------------------- 特征匹配 -----------------------------
def rotate_bound(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh)), M


def to33(M):
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        return M
    return np.vstack([M, [0, 0, 1.0]])


def prep_gray(img, match_size):
    h, w = img.shape[:2]
    s = match_size / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img.copy()
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    return g, s, small


def rootsift(desc):
    if desc is None:
        return None
    d = desc.astype(np.float32)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-7)
    return np.sqrt(d)


_SIFT = None


def sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=40000, contrastThreshold=0.008, edgeThreshold=24)
    return _SIFT


def tilt_simulate_M(shape, yaw_deg, tau_deg):
    """正射 crop 沿视线方位压缩 cos(tau) 的仿射矩阵(2x3)"""
    psi = math.radians(yaw_deg)
    t = max(0.25, math.cos(math.radians(tau_deg)))
    h, w = shape[:2]
    a, b = math.cos(psi), math.sin(psi)
    Rp = np.array([[a, -b], [b, a]])
    Rm = np.array([[a, b], [-b, a]])
    S = np.array([[1.0, 0.0], [0.0, t]])
    A2 = Rm @ S @ Rp
    return np.hstack([A2, [[w / 2], [h / 2]] - A2 @ np.array([[w / 2], [h / 2]])])


def match_photo_to_ref(crop_gray, crop_small, s1, pho_path, meta, log):
    """
    正射crop(灰度) vs 单张照片 匹配.
    返回 (M_ref2pho_3x3 [crop全图像素->照片全图像素], inliers, scale_pho_px_per_crop_px) 或 None
    """
    img_pho = imread_cn(pho_path)
    if img_pho is None:
        return None
    h, w = img_pho.shape[:2]
    if meta is not None:
        yaw = meta['GimbalYawDegree']
        tau = 90.0 + meta['GimbalPitchDegree']
        pho_north, Mrot = rotate_bound(img_pho, -yaw)
    else:
        # 无云台信息(被转码过的照片): 不预旋转, 依赖特征旋转不变性
        yaw, tau = 0.0, 0.0
        pho_north = img_pho
        Mrot = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    g2, s2, sm2 = prep_gray(pho_north, MATCH_SIZE)
    mask2 = cv2.threshold(cv2.cvtColor(sm2, cv2.COLOR_BGR2GRAY), 5, 255, cv2.THRESH_BINARY)[1]
    kp2, d2 = sift().detectAndCompute(g2, mask2)
    d2 = rootsift(d2)
    if d2 is None or len(kp2) < 30:
        return None

    attempts = []
    if meta is not None and 25 < tau <= 62:
        Mtilt = tilt_simulate_M(crop_gray.shape, yaw, tau)
        warped = cv2.warpAffine(crop_small, Mtilt, (crop_small.shape[1], crop_small.shape[0]))
        g1w = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
        attempts.append(('tiltsim', g1w, Mtilt))
    if tau <= 62:
        attempts.append(('direct', crop_gray, None))
    if not attempts:
        attempts.append(('direct', crop_gray, None))

    best = None
    for mode, g1, Mtilt in attempts:
        kp1, d1 = sift().detectAndCompute(g1, None)
        d1 = rootsift(d1)
        if d1 is None or len(kp1) < 30:
            continue
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
        good = [m for m, n in (pair for pair in knn if len(pair) == 2)
                if m.distance < 0.85 * n.distance]
        if len(good) < 10:
            continue
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 部分仿射(相似)与单应各试一次, 取内点多的; 剔除退化(近奇异)变换
        cand = []
        A, inlA = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0, maxIters=10000)
        if A is not None and inlA is not None:
            det = np.linalg.det(A[:, :2])
            if math.isfinite(det) and abs(det) > 1e-6:
                cand.append((to33(A), int(inlA.sum())))
        H, inlH = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=10000)
        if H is not None and inlH is not None:
            det = np.linalg.det(H[:2, :2])
            hn = abs(H[2, 2])
            if math.isfinite(det) and abs(det) > 1e-9 and hn > 1e-9:
                cand.append((to33(H), int(inlH.sum())))
        if not cand:
            continue
        M, ninl = max(cand, key=lambda t: t[1])
        # 组装链(右侧先作用):
        # crop全图 --S(s1)--> crop小图 --Mtilt(仅tiltsim)--> 变形小图
        #   --M--> 北向照片匹配尺度 --S(1/s2)--> 北向照片全图 --Minv--> 原照片全图
        S1 = np.array([[s1, 0, 0], [0, s1, 0], [0, 0, 1.0]])
        S2i = np.array([[1 / s2, 0, 0], [0, 1 / s2, 0], [0, 0, 1.0]])
        chain = to33(cv2.invertAffineTransform(Mrot)) @ S2i @ to33(M)
        if Mtilt is not None:
            chain = chain @ to33(Mtilt)
        chain = chain @ S1
        # 尺度 (照片全图像素 / crop全图像素), tiltsim 的压缩需剔除
        sc = math.sqrt(abs(np.linalg.det(chain[:2, :2])))
        if Mtilt is not None:
            sc /= math.sqrt(abs(np.linalg.det(Mtilt[:, :2])))
        if not math.isfinite(sc) or sc < 0.05 or sc > 100:
            log(f'    匹配[{mode}] 内点={ninl} 尺度异常({sc:.3f}), 丢弃')
            continue
        log(f'    匹配[{mode}] 内点={ninl} 尺度={sc:.3f}')
        if best is None or ninl > best[1]:
            best = (chain, ninl, sc, mode)
    return best


# ----------------------------- 配准 -----------------------------
def compute_registration(cands, img_ref_shape, metas, sizes, elev, lat0, lon0, log):
    """
    逐个候选匹配尝试: 用相机模型做几何门控, 找到可信锚点后建立
    (E,S)米 -> 正射全图像素 的相似配准; 含地面高程精化.
    cands: {photo_path: (chain3x3, ninl, scale, mode)}  chain: crop全图->照片全图
    返回 (A_geo_3x3, rho 正射像素/米, used_paths, elev_used)
    """
    x0, y0 = compute_registration.crop_xy
    per_cand = {}
    for path, (chain, ninl, sc, mode) in cands.items():
        meta = metas[path]
        w, h = sizes[path]
        f = eff_f_px(meta, w)
        tau = math.radians(90.0 + meta['GimbalPitchDegree'])
        gsd = meta['RelativeAltitude'] / f / (math.cos(tau) ** 1.5)   # 米/照片像素
        rho = 1.0 / (sc * gsd)      # 正射像素/米
        log(f'    候选 {os.path.basename(path)}: 内点={ninl} 估计尺度={rho:.2f}px/m '
            f'(变焦{meta.get("DigitalZoomRatio", 1.0):.2f}x)')
        if not math.isfinite(rho) or rho < 4 or rho > 90:
            log(f'      -> 尺度超出合理范围, 丢弃')
            continue
        inv = np.linalg.inv(chain)
        grid, dst = [], []
        for fu in (0.25, 0.5, 0.75):
            for fv in (0.25, 0.5, 0.75):
                refp = cv2.perspectiveTransform(np.array([[[w * fu, h * fv]]], np.float64),
                                                inv)[0, 0] + np.array([x0, y0])
                grid.append((w * fu, h * fv))
                dst.append(refp)
        per_cand[path] = dict(grid=grid, dst=np.float32(dst), ninl=ninl, rho=rho)
    if not per_cand:
        return None, None, [], elev

    def fit(paths, elev_use):
        src_l, dst_l = [], []
        for p in paths:
            d = per_cand[p]
            meta_p = metas[p]
            w, h = sizes[p]
            for (u, v), rp in zip(d['grid'], d['dst']):
                gpt = photo_px_to_ground(u, v, meta_p, w, h, elev_use, lat0, lon0)
                if gpt is None:
                    continue
                src_l.append([gpt[0], -gpt[1]])
                dst_l.append(rp)
        if len(src_l) < 5:
            return None, None, None
        src = np.float32(src_l)
        dst = np.float32(dst_l)
        A2, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if A2 is None:
            return None, None, None
        A_geo = to33(A2)
        rho_fit = math.sqrt(abs(np.linalg.det(A_geo[:2, :2])))
        if rho_fit < 1e-6:
            return None, None, None
        res = {}
        for p in paths:
            meta_p = metas[p]
            w, h = sizes[p]
            ax = axis_ground_point(meta_p, w, h, elev_use, lat0, lon0)
            if ax is None:
                res[p] = 999.0
                continue
            p_model = cv2.perspectiveTransform(np.array([[[ax[0], -ax[1]]]], np.float64),
                                               A_geo)[0, 0]
            p_match = cv2.perspectiveTransform(
                np.array([[[w / 2, h / 2]]], np.float64),
                np.linalg.inv(cands[p][0]))[0, 0] + np.array([x0, y0])
            res[p] = float(np.linalg.norm(p_model - p_match)) / rho_fit
        return A_geo, rho_fit, res

    # 按内点降序逐个尝试锚点; 低内点候选需通过更严的几何门控
    ordered = sorted(per_cand, key=lambda p: -per_cand[p]['ninl'])
    primary = None
    for p in ordered:
        ninl = per_cand[p]['ninl']
        if ninl < 15:
            continue
        gate = MODEL_GATE_M if ninl >= 25 else 3.0
        A_geo, rho_fit, res = fit([p], elev)
        if A_geo is None:
            continue
        log(f'    锚点试 {os.path.basename(p)}: 内点={ninl} '
            f'拟合尺度={rho_fit:.2f}px/m 几何残差={res[p]:.1f}m (门限{gate:.0f}m)')
        if res[p] <= gate and 4 <= rho_fit <= 90:
            primary = (p, A_geo, rho_fit, res[p])
            break
        log(f'      -> 未通过几何门控, 换下一个候选')
    if primary is None:
        return None, None, [], elev

    p0, A_geo, rho_fit, res0 = primary
    # 高程精化: 倾斜锚点的光轴地面点位置随高程变化, 可解算真实地面高程.
    # 严格保护: 大倾角 + 显著改善 + 内部极小值, 防止近垂直锚点的尺度简并被误用
    elev_use = elev
    tau0 = 90.0 + metas[p0]['GimbalPitchDegree']
    if tau0 > 20:
        scan = []
        for de in range(-40, 41, 2):
            A2d, rhod, resd = fit([p0], elev + de)
            if A2d is not None:
                scan.append((de, resd[p0]))
        if scan:
            best_de, best_res = min(scan, key=lambda t: t[1])
            interior = abs(best_de) < 38
            convex = all(r > best_res + 0.3 for de, r in scan if abs(de - best_de) in (2, 4))
            improved = best_res < 0.6 * res0 and (res0 - best_res) > 1.0
            if interior and convex and improved:
                elev_use = elev + best_de
                log(f'    高程精化: 地面高程修正 {best_de:+d}m (残差 {res0:.1f}m -> {best_res:.1f}m)')
                A_geo, rho_fit, _ = fit([p0], elev_use)
            elif abs(best_de) >= 2:
                log(f'    高程精化结果不可信(修正量{best_de:+d}m触及边界或改善不足), 保持起飞点高程')
    used = [p0]
    # 吸收其他与主锚点一致的候选
    for p in ordered:
        if p == p0 or per_cand[p]['ninl'] < 15:
            continue
        gate = MODEL_GATE_M if per_cand[p]['ninl'] >= 25 else 3.0
        A2g, rho2, res = fit(used + [p], elev_use)
        if A2g is None:
            continue
        if res[p] <= gate and abs(rho2 - rho_fit) / rho_fit < 0.2:
            used.append(p)
            A_geo, rho_fit = A2g, rho2
            log(f'    吸收候选 {os.path.basename(p)} (残差 {res[p]:.1f}m)')
    rot = math.degrees(math.atan2(A_geo[1, 0], A_geo[0, 0]))
    log(f'    配准完成: {rho_fit:.2f}px/m (GSD {100 / rho_fit:.1f}cm), '
        f'北向偏差 {rot:.2f}°, 锚点 {len(used)} 个')
    return A_geo, rho_fit, used, elev_use


# ----------------------------- 模型投影与精化 -----------------------------
def model_homography(A_geo, meta, w, h, elev, lat0, lon0, poly_ref_pts):
    """
    由相机模型计算 正射全图像素 -> 照片像素 的单应(平地假设).
    在红线周围取样本点: 正射px ->(E,S)->(E,N)->照片px
    """
    A_inv = np.linalg.inv(A_geo)
    pts = np.asarray(poly_ref_pts, np.float64).reshape(-1, 2)
    x, y, bw, bh = cv2.boundingRect(pts.astype(np.float32).reshape(-1, 1, 2))
    xs = np.linspace(x - bw * 0.6, x + bw * 1.6, 6)
    ys = np.linspace(y - bh * 0.6, y + bh * 1.6, 6)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    src, dst = [], []
    for px, py in grid:
        es = cv2.perspectiveTransform(np.array([[[px, py]]], np.float64), A_inv)[0, 0]
        uv = ground_to_photo_px(es[0], -es[1], meta, w, h, elev, lat0, lon0)
        if uv is None:
            continue
        src.append([px, py])
        dst.append(uv)
    if len(src) < 8:
        return None
    Hm, _ = cv2.findHomography(np.float32(src), np.float32(dst), method=0)
    return Hm


def refine_by_ncc(img_ref_gray, north_gray, H_ref2north_s, polys, rho, qscale,
                  M33_north=None, log=lambda s: None):
    """
    沿红线取补丁做模板匹配, 三级精化使红线在照片中更细致贴合:
      A. 整体平移(中位数, 抗离群);
      B. 相似修正(平移+微小旋转/缩放, 峰值加权最小二乘, 修正模型姿态小误差);
      C. 局部残差位移场 -> 沿红线加密点的高斯加权平滑微调(非刚性贴合局部纹理).
    img_ref_gray: 正射全图灰度;  north_gray: 北向上照片灰度(已按 qscale 缩小);
    H_ref2north_s: 正射全图像素 -> 北向上缩小图像素 的单应;
    rho: 正射像素/米;  qscale: 缩小图/全尺寸北向图 比例;
    M33_north: 原照片像素 -> 北向全尺寸 的仿射(供 warp 坐标往返).
    返回 dict(sim=北向全尺寸3x3修正阵, warp=callable|None, n_used, shift_m); 失败返回 None.
    warp 接收"原照片像素"坐标的多边形(Nx1x2), 输出加密并贴合后的同形多边形.
    """
    hh, ww = north_gray.shape
    # 沿多边形均匀采样(正射像素坐标), 采样更密以估计更细致的位移场
    samples = []
    for poly in polys:
        pts = poly.reshape(-1, 2).astype(np.float64)
        per = cv2.arcLength(poly, True)
        n = max(10, int(per / 80))
        for i in range(len(pts)):
            p0, p1 = pts[i], pts[(i + 1) % len(pts)]
            m = max(2, n // len(pts) + 1)
            for t in np.linspace(0, 1, m, endpoint=False):
                samples.append(p0 * (1 - t) + p1 * t)
    if not samples:
        return None
    half = NCC_PATCH // 2
    Hh, Hw = img_ref_gray.shape
    preds, founds, peaks, locals_ = [], [], [], []
    for sp in samples:
        x, y = int(sp[0]), int(sp[1])
        if x - half < 0 or y - half < 0 or x + half >= Hw or y + half >= Hh:
            continue
        tpl = img_ref_gray[y - half:y + half, x - half:x + half]
        if tpl.std() < 8:      # 无纹理补丁跳过
            continue
        # 局部尺度: 正射px -> 北向全尺寸照片px
        p0 = cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), H_ref2north_s)[0, 0] / qscale
        p1 = cv2.perspectiveTransform(np.array([[[x + 10, y]]], np.float64), H_ref2north_s)[0, 0] / qscale
        local = np.linalg.norm(p1 - p0) / 10.0          # 北向全尺寸照片px / 正射px
        if local < 0.15 or local > 40:
            continue
        neww = max(14, int(NCC_PATCH * local * qscale))  # 缩小图上的模板边长
        if neww > min(hh, ww) - 8:
            continue
        tpl_s = cv2.resize(tpl, (neww, neww))
        R = int(NCC_SEARCH_M * rho * local * qscale)     # 缩小图上的搜索半径
        R = max(24, min(R, 500))
        cx, cy = int(p0[0] * qscale), int(p0[1] * qscale)
        x0s, x1s = max(0, cx - neww // 2 - R), min(ww, cx + neww // 2 + R)
        y0s, y1s = max(0, cy - neww // 2 - R), min(hh, cy + neww // 2 + R)
        if x1s - x0s < neww + 4 or y1s - y0s < neww + 4:
            continue
        win = north_gray[y0s:y1s, x0s:x1s]
        if win.std() < 6:       # 旋转黑角等无效区域
            continue
        res = cv2.matchTemplate(win, tpl_s, cv2.TM_CCOEFF_NORMED)
        _, peak, _, mloc = cv2.minMaxLoc(res)
        if peak < NCC_MIN_PEAK:
            continue
        mx, my = mloc
        found_c = np.array([x0s + mx + neww / 2, y0s + my + neww / 2]) / qscale
        preds.append(np.array([p0[0], p0[1]]))            # 北向全尺寸像素
        founds.append(found_c)
        peaks.append(float(peak))
        locals_.append(local)
    if len(preds) < NCC_MIN_PATCHES:
        return None
    P = np.array(preds)
    F = np.array(founds)
    Wpk = np.array(peaks)
    med_local = float(np.median(locals_))
    # A. 整体平移 + 一致性内点筛选
    med = np.median(F - P, axis=0)
    dist = np.linalg.norm((F - P) - med, axis=1)
    thr = 2.5 * rho * med_local                           # 2.5米一致性阈值
    inl = dist < max(thr, 8.0)
    if inl.sum() < NCC_MIN_PATCHES:
        return None
    P, F, Wpk = P[inl], F[inl], Wpk[inl]
    n_used = int(inl.sum())
    # B. 峰值加权相似最小二乘: found = s*R(theta)*pred + t, 仅允许微小修正
    px, py = P[:, 0], P[:, 1]
    Amat = np.zeros((2 * len(P), 4))
    Amat[0::2] = np.stack([px, -py, np.ones_like(px), np.zeros_like(px)], axis=1)
    Amat[1::2] = np.stack([py, px, np.zeros_like(py), np.ones_like(py)], axis=1)
    bvec = F.reshape(-1)
    wrow = np.repeat(Wpk, 2)
    sol, *_ = np.linalg.lstsq(Amat * wrow[:, None], bvec * wrow, rcond=None)
    a_, b_ = float(sol[0]), float(sol[1])
    s_sim = math.hypot(a_, b_)
    theta = math.degrees(math.atan2(b_, a_))
    if abs(theta) <= 3.0 and 0.94 <= s_sim <= 1.06:
        sim = np.array([[a_, -b_, sol[2]], [b_, a_, sol[3]], [0.0, 0.0, 1.0]])
        log(f'    精化B级: 旋转{theta:+.2f}° 缩放{s_sim:.3f}')
    else:
        # 修正量过大不可信, 退回纯平移
        t0 = np.median(F - P, axis=0)
        sim = np.array([[1.0, 0.0, t0[0]], [0.0, 1.0, t0[1]], [0.0, 0.0, 1.0]])
        a_, b_ = 1.0, 0.0
    tnorm = math.hypot(float(sim[0, 2]), float(sim[1, 2]))
    shift_m = tnorm / (rho * med_local)
    if shift_m > NCC_MAX_SHIFT_M:
        return None
    # C. 相似修正后的残差位移场 -> 沿红线的平滑非刚性微调
    warp = None
    Fhat = (sim[:2, :2] @ P.T).T + sim[:2, 2]
    resid = F - Fhat                                      # 北向全尺寸像素残差
    if M33_north is not None and len(P) >= 5:
        res_norm = np.linalg.norm(resid, axis=1)
        cap_px = 2.0 * rho * med_local                    # 单点最大微调 2 米
        radius = max(200.0, 2.5 * float(np.median(
            np.linalg.norm(np.diff(P, axis=0), axis=0)) if len(P) > 1 else 200.0))
        centers = Fhat.copy()
        rw = Wpk.copy()
        Minv = np.linalg.inv(M33_north)

        def warp(poly_pho):
            pts = cv2.perspectiveTransform(poly_pho.astype(np.float64),
                                           M33_north).reshape(-1, 2)
            # 沿红线加密(步长 ~ 补丁采样间距), 使贴合更细致
            step = max(30.0, radius / 3.0)
            dens = []
            for i in range(len(pts)):
                pa, pb = pts[i], pts[(i + 1) % len(pts)]
                L = float(np.linalg.norm(pb - pa))
                m = max(1, int(L / step))
                for k2 in range(m):
                    dens.append(pa * (1 - k2 / m) + pb * (k2 / m))
            dens = np.array(dens)
            off = np.zeros_like(dens)
            for s0 in range(0, len(dens), 400):
                chunk = dens[s0:s0 + 400]
                D = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2)
                w = rw[None, :] * np.exp(-(D / radius) ** 2)
                w[D > 2 * radius] = 0.0
                ws = w.sum(axis=1)
                ok = ws > 1e-6
                off[s0:s0 + 400][ok] = (w[ok] @ resid) / ws[ok, None]
            # 限幅 + 沿环滑动平均(保形, 防抖动)
            on = np.linalg.norm(off, axis=1)
            big = on > cap_px
            off[big] *= (cap_px / on[big])[:, None]
            kwin = 5
            pad = np.concatenate([off[-kwin:], off, off[:kwin]])
            ker = np.ones(2 * kwin + 1) / (2 * kwin + 1)
            off = np.stack([np.convolve(pad[:, 0], ker, 'valid'),
                            np.convolve(pad[:, 1], ker, 'valid')], axis=1)
            warped_n = dens + off
            back = cv2.perspectiveTransform(warped_n.reshape(-1, 1, 2), Minv)
            if _poly_self_intersects(back.reshape(-1, 2)):
                return poly_pho
            return back
    return dict(sim=sim, warp=warp, n_used=n_used, shift_m=shift_m)


def apply_translation(H, dx, dy):
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    return T @ H


# ----------------------------- 绘制与保存 -----------------------------
def draw_redlines(img_pho, polys_pho, color=(0, 0, 255)):
    out = img_pho.copy()
    h, w = out.shape[:2]
    thick = max(3, min(h, w) // 300)
    for pts in polys_pho:
        cv2.polylines(out, [pts.astype(np.int32)], True, color, thick, cv2.LINE_AA)
    return out


def frame_visibility(polys_pho, w, h):
    """栅格化判断红线与照片画面的关系.
    返回 (line_vis 线是否穿过画面, coverage 画面被红线区域覆盖的比例)"""
    sc = 500.0 / max(w, h)
    cw, ch = int(w * sc) + 4, int(h * sc) + 4
    outline = np.zeros((ch, cw), np.uint8)
    fill = np.zeros((ch, cw), np.uint8)
    for poly in polys_pho:
        pts = np.clip(poly.reshape(-1, 2), -1e6, 1e6)
        p = (pts * sc).astype(np.int32)
        if len(p) >= 2:
            cv2.polylines(outline, [p], True, 1, 2)
        if len(p) >= 3:
            cv2.fillPoly(fill, [p], 1)
    fh, fw = int(h * sc), int(w * sc)
    frame = np.zeros((ch, cw), bool)
    frame[:fh, :fw] = True
    line_vis = bool(((outline > 0) & frame).any())
    coverage = float(fill[:fh, :fw].mean())
    return line_vis, coverage


_CJK_FONT = {}


def draw_inside_mark(img_pho, text='红线内部照片'):
    """整张照片在红线内部: 四周画红框, 左下角红色小字标注"""
    out = img_pho.copy()
    h, w = out.shape[:2]
    m = int(min(h, w) * 0.018)
    thick = max(3, min(h, w) // 300)
    cv2.rectangle(out, (m, m), (w - m, h - m), (0, 0, 255), thick, cv2.LINE_AA)
    try:
        from PIL import Image, ImageDraw, ImageFont
        size = max(16, int(min(h, w) * 0.026))          # 偏小的字号
        if size not in _CJK_FONT:
            font = None
            for fp in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
                       r'C:\Windows\Fonts\simsun.ttc', r'C:\Windows\Fonts\Deng.ttf'):
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, size)
                    break
            _CJK_FONT[size] = font or ImageFont.load_default()
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        tb = dr.textbbox((0, 0), text, font=_CJK_FONT[size])
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        dr.text((m + thick + 12, h - m - thick - th - 14), text,
                font=_CJK_FONT[size], fill=(255, 0, 0))
        out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(out, text, (m + 12, h - m - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


def save_with_exif(src_path, dst_path, img, quality=92, keep_exif=True):
    ext = os.path.splitext(dst_path)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp'):
        dst_path += '.jpg'
        ext = '.jpg'
    ok = imwrite_cn(dst_path, img, quality)
    if not ok:
        return dst_path, False
    if keep_exif and ext in ('.jpg', '.jpeg'):
        try:
            import piexif
            exif_bytes = piexif.dump(piexif.load(src_path))
            piexif.insert(exif_bytes, dst_path)
        except Exception:
            pass
    return dst_path, True


# ----------------------------- 单地块处理 -----------------------------
def process_plot(pid, ref_path, photo_list, out_dir, opts, log):
    """
    返回 (report_rows, contact_sheet_img or None)
    """
    rows = []
    log(f'【{pid}】正射图: {os.path.basename(ref_path)}, 照片 {len(photo_list)} 张')
    img_ref = imread_cn(ref_path)
    if img_ref is None:
        log(f'【{pid}】正射图读取失败, 跳过')
        for pp in photo_list:
            rows.append([pid, os.path.basename(pp), '失败', '正射图读取失败', '', '', '', ''])
        return rows, None

    # 1. 红线提取
    polys = extract_redlines(img_ref, log=lambda s: log(f'【{pid}】{s}'))
    if not polys:
        log(f'【{pid}】未检测到红线(可能未绘制或被屋顶干扰), 跳过')
        for pp in photo_list:
            rows.append([pid, os.path.basename(pp), '未处理', '正射图未检测到红线', '', '', '', ''])
        return rows, None
    log(f'【{pid}】提取红线 {len(polys)} 条, 顶点数: {[len(p[0]) for p in polys]}')
    poly_list = [p[0] for p in polys]
    all_pts = np.vstack([p.reshape(-1, 2) for p in poly_list])
    cxc, cyc = all_pts.mean(axis=0)

    # 2. 照片元数据
    metas, sizes = {}, {}
    photos_ok = []
    for pp in photo_list:
        meta = read_photo_meta(pp)
        img = imread_cn(pp)
        if img is None:
            rows.append([pid, os.path.basename(pp), '失败', '照片读取失败', '', '', '', ''])
            continue
        sizes[pp] = (img.shape[1], img.shape[0])
        if meta is None:
            log(f'  {os.path.basename(pp)}: 无定位/姿态信息')
        else:
            metas[pp] = meta
        photos_ok.append(pp)
        del img
    if not photos_ok:
        return rows, None

    # 地面高程估计: 各照片 (绝对-相对) 中位数
    elevs = [metas[p]['AbsoluteAltitude'] - metas[p]['RelativeAltitude'] for p in metas]
    elev = float(np.median(elevs)) if elevs else 0.0
    # 质心经纬度原点: 用第一张照片的 GPS 近似(仅用于米制换算, 不影响精度)
    lat0 = lon0 = None
    for p in metas:
        lat0, lon0 = metas[p]['GpsLatitude'], metas[p]['GpsLongitude']
        break
    if lat0 is None:
        lat0, lon0 = 29.0, 120.0     # 仅退化情况

    # 3. 特征匹配寻找锚点
    x, y, bw, bh = cv2.boundingRect(all_pts.astype(np.float32).reshape(-1, 1, 2))
    side = max(max(bw, bh) * 2.5, 900)       # 小地块也要给匹配留足上下文
    W, H = img_ref.shape[1], img_ref.shape[0]
    x0 = max(0, int(x + bw / 2 - side / 2)); y0 = max(0, int(y + bh / 2 - side / 2))
    x1 = min(W, int(x + bw / 2 + side / 2)); y1 = min(H, int(y + bh / 2 + side / 2))
    crop = img_ref[y0:y1, x0:x1]
    compute_registration.crop_xy = (x0, y0)
    crop_gray, s1, crop_small = prep_gray(crop, MATCH_SIZE)

    # 按离天底角升序尝试
    def tau_of(p):
        m = metas.get(p)
        return 90 + m['GimbalPitchDegree'] if m else 999
    ordered = sorted(photos_ok, key=tau_of)
    cands = {}
    for pp in ordered:
        meta_pp = metas.get(pp)
        log(f'  匹配 {os.path.basename(pp)} (倾角 {tau_of(pp):.0f}°'
            + ('' if meta_pp else ', 无定位信息') + ')...')
        try:
            r = match_photo_to_ref(crop_gray, crop_small, s1, pp, meta_pp,
                                   lambda s: log(f'  {os.path.basename(pp)}{s}'))
        except Exception as e:
            log(f'    匹配异常: {e}')
            r = None
        if r is not None and r[1] >= 12:
            cands[pp] = r
        if len(cands) >= 3 and tau_of(pp) > 40:
            break     # 已有足够候选, 不再尝试大倾角照片

    A_geo = None
    rho = None
    used = []
    reg_cands = {p: c for p, c in cands.items() if p in metas}
    if reg_cands:
        A_geo, rho, used, elev = compute_registration(reg_cands, img_ref.shape, metas, sizes,
                                                      elev, lat0, lon0,
                                                      lambda s: log(f'【{pid}】{s}'))
    if A_geo is None:
        log(f'【{pid}】特征配准失败, 尝试模板匹配引导配准...')
        A_geo, rho = ncc_bootstrap(img_ref, crop, (x0, y0), ordered, metas, sizes,
                                   elev, lat0, lon0, poly_list,
                                   lambda s: log(f'【{pid}】{s}'))
    # 无配准时, 强匹配(>=60内点)的无定位照片仍可直接转绘
    direct_ok = set()
    if A_geo is None:
        direct_ok = {p for p, c in cands.items() if c[1] >= 60}
        if not direct_ok:
            log(f'【{pid}】配准失败, 本地块无法自动转绘')
            for pp in photos_ok:
                rows.append([pid, os.path.basename(pp), '失败', '配准失败(纹理变化过大)', '', '', '', ''])
            return rows, None
        log(f'【{pid}】无配准参数, 仅对强匹配照片直接转绘')

    # 4. 逐照片转绘
    ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    out_plot_dir = os.path.join(out_dir, pid)
    os.makedirs(out_plot_dir, exist_ok=True)
    footprints = []
    for pp in photos_ok:
        name = os.path.basename(pp)
        meta = metas.get(pp)
        w, h = sizes[pp]
        method = ''
        H_ref2pho = None
        inl_cnt = ''
        if pp in used or (pp not in metas and pp in cands and cands[pp][1] >= 60):
            # 通过门控的匹配 / 无定位信息的强匹配: 链式变换 crop->photo, 补上 crop 偏移
            chain = cands[pp][0]
            T_off = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
            H_ref2pho = chain @ T_off
            method = '特征匹配' if pp in metas else '特征匹配(无定位信息)'
            inl_cnt = cands[pp][1]
        elif meta is not None and A_geo is not None:
            H_ref2pho = model_homography(A_geo, meta, w, h, elev, lat0, lon0, all_pts)
            method = '定位模型'
        if H_ref2pho is None:
            rows.append([pid, name, '失败', '无定位信息且匹配失败', '', '', '', ''])
            footprints.append((pp, None, 'fail'))
            continue

        # NCC 三级精化(平移+相似+局部贴合), 对所有带定位信息的照片启用;
        # 特征匹配锚点同样精化以消除模型姿态小误差
        shift_m = ''
        n_patch = ''
        warp_fn = None
        img_pho = imread_cn(pp)
        if meta is not None and rho is not None:
            try:
                pho_north, Mrot = rotate_bound(img_pho, -meta['GimbalYawDegree'])
                q = min(1.0, 4000 / max(pho_north.shape[:2]))
                north_small = cv2.resize(pho_north, None, fx=q, fy=q) if q < 1 else pho_north
                north_gray = cv2.cvtColor(north_small, cv2.COLOR_BGR2GRAY)
                del pho_north, north_small
                M33 = to33(Mrot)
                H_north = M33 @ H_ref2pho
                Sq = np.array([[q, 0, 0], [0, q, 0], [0, 0, 1.0]])
                rr = refine_by_ncc(ref_gray, north_gray, Sq @ H_north, poly_list, rho, q,
                                   M33_north=M33,
                                   log=lambda s: log(f'  {name}{s}'))
                del north_gray
                if rr is not None:
                    H_north = rr['sim'] @ H_north
                    H_ref2pho = np.linalg.inv(M33) @ H_north
                    warp_fn = rr['warp']
                    n_patch = rr['n_used']
                    shift_m = f"{rr['shift_m']:.1f}"
                    method += '+精化'
                    if warp_fn is not None:
                        method += '+贴合'
            except Exception:
                pass

        # 变换红线并判断可见性(栅格化: 线穿画面 / 画面被红线区域覆盖)
        polys_pho = []
        for poly in poly_list:
            pts = cv2.perspectiveTransform(poly.astype(np.float64), H_ref2pho)
            polys_pho.append(pts)
        if warp_fn is not None:
            polys_pho = [warp_fn(p) for p in polys_pho]
        line_vis, coverage = frame_visibility(polys_pho, w, h)
        is_inside = coverage >= 0.95
        if not is_inside and not line_vis and coverage < 0.02:
            # 红线不在画面: 原照片单独存档 + 生成 正射图-照片 位置对比图
            try:
                nf_dir = os.path.join(out_dir, '红线不在画面', pid)
                os.makedirs(nf_dir, exist_ok=True)
                shutil.copy2(pp, os.path.join(nf_dir, name))
                cmp_dir = os.path.join(out_dir, '红线不在画面', '对比图')
                os.makedirs(cmp_dir, exist_ok=True)
                cmp_img = make_out_comparison(img_ref, poly_list, H_ref2pho, img_pho, pid, name)
                cmp_path = os.path.join(cmp_dir, f'{pid}_{os.path.splitext(name)[0]}_对比.jpg')
                imwrite_cn(cmp_path, cmp_img, 90)
                log(f'  {name}: 红线不在画面, 已存档并生成对比图')
            except Exception as e:
                cmp_path = ''
                log(f'  {name}: 对比图生成失败 {e}')
            rows.append([pid, name, '红线不在画面(已单独存档)', method, inl_cnt, '', '', cmp_path])
            footprints.append((pp, H_ref2pho, 'out'))
            continue
        vis = max(coverage, 0.01) if (is_inside or line_vis) else 0.0
        if is_inside:
            out_img = draw_inside_mark(img_pho)
            method += '/整幅在红线内'
            log(f'  {name}: 整张照片位于红线内部, 已画边框标注')
        else:
            out_img = draw_redlines(img_pho, polys_pho)
        out_path = os.path.join(out_plot_dir, os.path.splitext(name)[0] + '_红线.jpg')
        out_path, ok = save_with_exif(pp, out_path, out_img,
                                      quality=opts.get('quality', 92),
                                      keep_exif=opts.get('keep_exif', True))
        conf = '高' if (pp in cands or n_patch != '') else '中'
        status = '成功' if conf == '高' else '成功(建议人工复核)'
        rows.append([pid, name, status, method, inl_cnt, n_patch, shift_m, out_path])
        footprints.append((pp, H_ref2pho, 'ok'))
        log(f'  {name}: {status} [{method}] 可见度{vis:.0%}')

    # 5. 对照图
    sheet = None
    if opts.get('make_preview', True):
        try:
            sheet = make_contact_sheet(img_ref, poly_list, footprints, sizes, pid, A_geo)
        except Exception as e:
            log(f'  对照图生成失败: {e}')
    return rows, sheet


def make_contact_sheet(img_ref, poly_list, footprints, sizes, pid, A_geo):
    """正射缩略图 + 红线(绿) + 各照片 footprint(黄=成功 青=模型 红=失败)"""
    vis = img_ref.copy()
    for poly in poly_list:
        cv2.polylines(vis, [poly], True, (0, 255, 0), 10, cv2.LINE_AA)
    Hh, Hw = vis.shape[:2]
    for pp, H_ref2pho, st in footprints:
        if H_ref2pho is None:
            continue
        w, h = sizes[pp]
        corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]).astype(np.float64)
        try:
            cref = cv2.perspectiveTransform(corners, np.linalg.inv(H_ref2pho))
        except Exception:
            continue
        color = {'ok': (0, 255, 255), 'out': (255, 0, 255), 'fail': (0, 0, 255)}.get(st, (255, 255, 0))
        cv2.polylines(vis, [cref.astype(np.int32)], True, color, 6, cv2.LINE_AA)
    s = 1600 / max(Hh, Hw)
    small = cv2.resize(vis, None, fx=s, fy=s)
    cv2.putText(small, f'Plot {pid}', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return small


def _put_cjk(img, text, org, size, color=(255, 0, 0)):
    """用系统中文字体写中文; 失败退化为英文"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        if size not in _CJK_FONT:
            font = None
            for fp in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
                       r'C:\Windows\Fonts\simsun.ttc', r'C:\Windows\Fonts\Deng.ttf'):
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, size)
                    break
            _CJK_FONT[size] = font or ImageFont.load_default()
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        dr.text(org, text, font=_CJK_FONT[size], fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img


def make_out_comparison(img_ref, poly_list, H_ref2pho, img_pho, pid, name):
    """红线不在画面: 正射图(绿=红线 品红=照片范围) 与照片的并排对比图"""
    w, h = img_pho.shape[1], img_pho.shape[0]
    vis = img_ref.copy()
    for poly in poly_list:
        cv2.polylines(vis, [poly], True, (0, 200, 0), 8, cv2.LINE_AA)
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float64)
    cref = cv2.perspectiveTransform(corners, np.linalg.inv(H_ref2pho)).reshape(-1, 2)
    cref_i = np.clip(cref, -1e6, 1e6).astype(np.int32)
    cv2.polylines(vis, [cref_i], True, (255, 0, 255), 8, cv2.LINE_AA)
    fc = cref.mean(axis=0).astype(int)
    cv2.drawMarker(vis, tuple(fc), (255, 0, 255), cv2.MARKER_STAR, 80, 8)
    Hh, Ww = vis.shape[:2]
    th = 1200
    left = cv2.resize(vis, None, fx=th / Hh, fy=th / Hh)
    right = cv2.resize(img_pho, None, fx=th / h, fy=th / h)
    # 顶部标题条
    title = f'编号{pid}  照片{name}: 品红框=照片拍摄范围, 绿线=红线 (该照片未覆盖红线)'
    bar = np.full((80, left.shape[1] + right.shape[1], 3), 255, np.uint8)
    out = np.vstack([bar, np.hstack([left, right])])
    out = _put_cjk(out, title, (18, 18), 40)
    out = _put_cjk(out, '无人机影像图', (18, 90), 34, (0, 150, 0))
    out = _put_cjk(out, '无人机照片', (left.shape[1] + 18, 90), 34, (0, 0, 200))
    return out


# ----------------------------- NCC 引导配准(回退) -----------------------------
def rectify_to_nadir(img, meta, max_out=3200):
    """纯旋转单应把照片矫正为北向上正射视角(同一投影中心, 对任意地形精确).
    返回 (矫正图, 有效焦距px, Ht 照片像素->矫正图像素)"""
    h, w = img.shape[:2]
    yaw_deg, pitch_deg = meta['GimbalYawDegree'], meta['GimbalPitchDegree']
    f_px = eff_f_px(meta, w)
    K = np.array([[f_px, 0, w / 2], [0, f_px, h / 2], [0, 0, 1.0]])
    r, v, d = camera_frames(yaw_deg, pitch_deg)
    R_wc = np.column_stack([r, v, d])
    R0_wc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    H = K @ (R0_wc.T @ R_wc) @ np.linalg.inv(K)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    c2 = cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2)
    mn, mx = c2.min(0), c2.max(0)
    ow, oh = mx - mn
    out_scale = min(1.0, max_out / max(ow, oh))
    T = np.array([[out_scale, 0, -mn[0] * out_scale],
                  [0, out_scale, -mn[1] * out_scale], [0, 0, 1.0]])
    Ht = T @ H
    out = cv2.warpPerspective(img, Ht, (int(ow * out_scale), int(oh * out_scale)))
    return out, f_px * out_scale, Ht


def ncc_bootstrap(img_ref, crop, crop_xy, ordered_photos, metas, sizes, elev, lat0, lon0,
                  poly_list, log):
    """当特征匹配全部失败时: 照片矫正为正射视角, 尺度搜索+边缘模板匹配 配准"""
    W, H = img_ref.shape[1], img_ref.shape[0]
    o_s = min(1.0, 3200 / max(W, H))
    ref_small = cv2.resize(img_ref, None, fx=o_s, fy=o_s) if o_s < 1 else img_ref
    eg_ref = cv2.Canny(cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY), 40, 120).astype(np.float32)
    # 屏蔽图幅边框/标题栏, 防止模板锁到图框上
    eh, ew = eg_ref.shape
    bm = int(0.02 * min(eh, ew))
    eg_ref[:bm, :] = 0; eg_ref[-bm:, :] = 0; eg_ref[:, :bm] = 0; eg_ref[:, -bm:] = 0
    eg_ref[:int(0.055 * eh), :] = 0

    for pp in ordered_photos:
        meta = metas.get(pp)
        if meta is None:
            continue
        tau = 90 + meta['GimbalPitchDegree']
        if tau > 58:
            continue
        img_pho = imread_cn(pp)
        if img_pho is None:
            continue
        w, h = sizes[pp]
        rect, f_rect, Ht = rectify_to_nadir(img_pho, meta)
        gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        # 有效内容最大内接矩形(去除旋转黑角)
        valid = (gray > 5).astype(np.uint8) * 255
        valid = cv2.morphologyEx(valid, cv2.MORPH_ERODE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        vx, vy, vw, vh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        shrink_x, shrink_y = int(vw * 0.08), int(vh * 0.08)
        vx += shrink_x; vy += shrink_y; vw -= 2 * shrink_x; vh -= 2 * shrink_y
        if vw < 200 or vh < 200:
            continue
        tpl_img = gray[vy:vy + vh, vx:vx + vw]
        eg_pho = cv2.Canny(tpl_img, 40, 120).astype(np.float32)
        # 模板中心 -> 矫正图坐标 -> 照片像素 -> 地面点(精确)
        tc_rect = np.array([[[vx + vw / 2, vy + vh / 2]]], np.float64)
        tc_pho = cv2.perspectiveTransform(tc_rect, np.linalg.inv(Ht))[0, 0]
        anchor_g = photo_px_to_ground(tc_pho[0], tc_pho[1], meta, w, h, elev, lat0, lon0)
        if anchor_g is None:
            continue
        gsd_rect = meta['RelativeAltitude'] / f_rect     # 米/矫正像素(星下点附近)
        best = None
        for rho in np.geomspace(6, 45, 20):              # 正射 px/米 候选
            sc = rho * gsd_rect * o_s                    # 缩小正射px / 矫正px
            tw, th = int(vw * sc), int(vh * sc)
            if tw < 40 or th < 40 or tw >= ew - 2 or th >= eh - 2:
                continue
            tpl = cv2.resize(eg_pho, (tw, th))
            res = cv2.matchTemplate(eg_ref, tpl, cv2.TM_CCOEFF_NORMED)
            _, peak, _, mloc = cv2.minMaxLoc(res)
            if best is None or peak > best[0]:
                best = (peak, rho, mloc, tw, th)
        if best is None or best[0] < 0.30:
            log(f'  引导配准 {os.path.basename(pp)}: 最佳相关度不足'
                + ('' if best is None else f'({best[0]:.2f})'))
            continue
        peak, rho, (mx, my), tw, th = best
        log(f'  引导配准 {os.path.basename(pp)}: 相关度={peak:.2f} 尺度={rho:.1f}px/m')
        # 模板中心(=anchor_g地面点)对应的正射全图像素
        tpl_c = np.array([(mx + tw / 2) / o_s, (my + th / 2) / o_s])
        # A_geo: (E,S=-N) -> 正射px, scale=rho, rot=0, 平移使 anchor->tpl_c
        A2 = np.array([[rho, 0, tpl_c[0] - rho * anchor_g[0]],
                       [0, rho, tpl_c[1] - rho * (-anchor_g[1])]], np.float64)
        log(f'  引导配准成功(低置信, 建议人工复核对照图)')
        return to33(A2), rho
    return None, None


# ----------------------------- 批处理 -----------------------------
def read_ids_from_xlsx(path, log=print):
    """从 Excel 表格读取要处理的编号(任意单元格中的纯数字均可识别, 位数不限)"""
    import openpyxl
    ids = []
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for v in row:
                if v is None:
                    continue
                s = str(v).strip()
                if s.endswith('.0'):
                    s = s[:-2]
                if re.fullmatch(r'\d+', s):
                    ids.append(s)
    wb.close()
    ids = sorted(set(ids))
    log(f'从 Excel 读取到 {len(ids)} 个编号: {", ".join(ids) if ids else "(空)"}')
    return ids


def write_ids_template(path, ids):
    """生成编号模板 Excel"""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '处理编号'
    ws.append(['编号'])
    for pid in ids:
        ws.append([pid])
    ws.column_dimensions['A'].width = 16
    wb.save(path)


def id_candidates(name):
    """从名称提取候选编号: XXXXX号(图斑) 中的数字优先(位数不限), 然后其余数字串; 年份不算"""
    m = re.search(r'(\d+)\s*号', name)
    if m:
        return [m.group(1)]
    return _digits_not_year(name)


def scan_inputs(ref_dir, photo_dir, log):
    """扫描输入, 返回 {pid: (ref_path, [photo_paths])}, 以及警告列表.
    文件夹名任意: 正射图编号与照片(文件名或上级文件夹)编号一致即配对"""
    refs = {}
    for f in sorted(os.listdir(ref_dir)):
        if os.path.splitext(f)[1].lower() not in IMG_EXTS:
            continue
        pid = get_plot_id(f)
        if pid is None:
            log(f'警告: 正射图 {f} 未识别编号, 忽略')
            continue
        refs[pid] = os.path.join(ref_dir, f)
    photos = {}
    for root, _, files in os.walk(photo_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in IMG_EXTS:
                continue
            # 候选编号: 文件名优先, 其次上级文件夹名; 多个候选时优先能与正射图配对的
            cand = id_candidates(f) + id_candidates(os.path.basename(root))
            pid = next((c for c in cand if c in refs), cand[0] if cand else None)
            if pid is None:
                continue
            photos.setdefault(pid, []).append(os.path.join(root, f))
    plots = {}
    for pid in refs:
        if pid in photos:
            plots[pid] = (refs[pid], sorted(photos[pid]))
    return plots, refs, photos


def scan_photos_all(photo_dir):
    """递归收集照片文件夹下所有照片(单图模式用)"""
    out = []
    for root, _, files in os.walk(photo_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def run_batch(ref_dir, photo_dir, out_dir, opts, log, progress=None, cancel=None):
    plots, refs, photos = scan_inputs(ref_dir, photo_dir, log)
    only = set(refs) - set(photos)
    for pid in sorted(only):
        log(f'提示: 编号 {pid} 只有正射图没有照片, 跳过')
    for pid in sorted(set(photos) - set(refs)):
        log(f'提示: 编号 {pid} 只有照片没有正射图, 跳过 ({len(photos[pid])} 张照片)')
    if not plots:
        log('没有找到编号匹配的正射图-照片组合!')
        return [], 0, 0

    os.makedirs(out_dir, exist_ok=True)
    rows_all = []
    pids = sorted(plots)
    only_ids = opts.get('only_ids') or []
    if opts.get('ids_file'):
        try:
            only_ids = read_ids_from_xlsx(opts['ids_file'], log)
        except Exception as e:
            log(f'读取编号 Excel 失败: {e}, 改为处理全部编号')
            only_ids = []
    if only_ids:
        missing = [p for p in only_ids if p not in plots]
        for p in missing:
            log(f'提示: 编号 {p} 在输入文件夹中找不到匹配的正射图+照片组合')
        pids = [p for p in pids if p in only_ids]
        log(f'按指定编号处理 {len(pids)} 个地块')
    n_done = 0
    prev_dir = os.path.join(out_dir, '对照预览')
    if opts.get('make_preview', True):
        os.makedirs(prev_dir, exist_ok=True)
    for i, pid in enumerate(pids):
        if cancel is not None and cancel.is_set():
            log('已取消')
            break
        log(f'===== [{i + 1}/{len(pids)}] 编号 {pid} =====')
        try:
            rows, sheet = process_plot(pid, plots[pid][0], plots[pid][1],
                                       out_dir, opts, log)
            rows_all.extend(rows)
            if sheet is not None:
                imwrite_cn(os.path.join(prev_dir, f'{pid}_对照图.jpg'), sheet)
        except Exception as e:
            log(f'【{pid}】处理异常: {e}')
            log(traceback.format_exc(limit=3))
            for pp in plots[pid][1]:
                rows_all.append([pid, os.path.basename(pp), '失败', f'异常:{e}', '', '', '', ''])
        n_done += 1
        if progress:
            progress((i + 1) / len(pids))

    # 报告
    rep = os.path.join(out_dir, '处理报告.csv')
    with open(rep, 'w', newline='', encoding='utf-8-sig') as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(['编号', '照片', '状态', '方式', '匹配内点', '精化补丁数', '精化位移', '输出路径'])
        wcsv.writerows(rows_all)
    n_ok = sum(1 for r in rows_all if str(r[2]).startswith('成功'))
    n_fail = sum(1 for r in rows_all if r[2] == '失败')
    n_skip = len(rows_all) - n_ok - n_fail
    log(f'===== 完成: 成功 {n_ok}, 失败 {n_fail}, 未处理/不在画面 {n_skip} =====')
    log(f'报告: {rep}')
    return rows_all, n_ok, n_fail


def run_single(ref_file, photo_dir, out_dir, opts, log, progress=None, cancel=None):
    """单图模式: 一张正射影像图 对 一个照片文件夹(不要求编号一致)"""
    if not os.path.isfile(ref_file):
        log(f'正射影像图文件不存在: {ref_file}')
        return [], 0, 0
    photos = scan_photos_all(photo_dir)
    if not photos:
        log(f'照片文件夹中没有照片: {photo_dir}')
        return [], 0, 0
    pid = get_plot_id(os.path.basename(ref_file)) or '单图'
    log(f'单图模式: 影像图 {os.path.basename(ref_file)}, 照片 {len(photos)} 张')
    os.makedirs(out_dir, exist_ok=True)
    try:
        rows, sheet = process_plot(pid, ref_file, photos, out_dir, opts, log)
    except Exception as e:
        log(f'处理异常: {e}')
        log(traceback.format_exc(limit=3))
        rows = [[pid, os.path.basename(pp), '失败', f'异常:{e}', '', '', '', ''] for pp in photos]
    if sheet is not None and opts.get('make_preview', True):
        prev_dir = os.path.join(out_dir, '对照预览')
        os.makedirs(prev_dir, exist_ok=True)
        imwrite_cn(os.path.join(prev_dir, f'{pid}_对照图.jpg'), sheet)
    rep = os.path.join(out_dir, '处理报告.csv')
    with open(rep, 'w', newline='', encoding='utf-8-sig') as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(['编号', '照片', '状态', '方式', '匹配内点', '精化补丁数', '精化位移', '输出路径'])
        wcsv.writerows(rows)
    n_ok = sum(1 for r in rows if str(r[2]).startswith('成功'))
    log(f'===== 完成: 成功 {n_ok}, 其他 {len(rows) - n_ok} =====')
    log(f'报告: {rep}')
    if progress:
        progress(1.0)
    return rows, n_ok, len(rows) - n_ok


# ----------------------------- GUI -----------------------------
def build_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext, messagebox

    root = tk.Tk()
    root.title('无人机照片红线自动匹配工具 V1.3')
    try:
        root.iconbitmap(default=_resource_path(os.path.join('assets', 'icon.ico')))
    except Exception:
        pass
    root.geometry('820x700')

    base = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, 'frozen', False) else __file__))

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='x')

    # 模式选择
    mode_var = tk.StringVar(value='batch')
    ttk.Radiobutton(frm, text='批量模式: 多张影像图按编号自动配对(文件夹名任意)',
                    variable=mode_var, value='batch',
                    command=lambda: on_mode()).grid(row=0, column=0, columnspan=3, sticky='w')
    ttk.Radiobutton(frm, text='单图模式: 一张影像图 对 一个照片文件夹(不要求编号一致)',
                    variable=mode_var, value='single',
                    command=lambda: on_mode()).grid(row=1, column=0, columnspan=3, sticky='w')

    vars_ = {}
    lbl_ref = ttk.Label(frm, text='正射影像图文件夹')
    lbl_ref.grid(row=2, column=0, sticky='w', pady=3)
    # ref 行
    vars_['ref'] = tk.StringVar(value=os.path.join(base, '无人机正射影像图'))
    ttk.Entry(frm, textvariable=vars_['ref'], width=64).grid(row=2, column=1, padx=5)
    btn_ref = ttk.Button(frm, text='浏览...', command=lambda: None)
    btn_ref.grid(row=2, column=2)
    # photo / out 行
    for i, (key, label, dft) in enumerate([('photo', '无人机照片文件夹', '无人机照片'),
                                           ('out', '输出文件夹', '红线输出')], start=3):
        ttk.Label(frm, text=label).grid(row=i, column=0, sticky='w', pady=3)
        v = tk.StringVar(value=os.path.join(base, dft))
        vars_[key] = v
        ttk.Entry(frm, textvariable=v, width=64).grid(row=i, column=1, padx=5)
        ttk.Button(frm, text='浏览...',
                   command=lambda v=v: v.set(filedialog.askdirectory() or v.get())
                   ).grid(row=i, column=2)

    def browse_ref():
        if mode_var.get() == 'single':
            p = filedialog.askopenfilename(
                filetypes=[('影像图', '*.jpg *.jpeg *.tif *.tiff *.png *.bmp')])
            if p:
                vars_['ref'].set(p)
        else:
            p = filedialog.askdirectory()
            if p:
                vars_['ref'].set(p)
    btn_ref.configure(command=browse_ref)

    # 编号 Excel 行
    v_xlsx = tk.StringVar(value='')
    lbl_xlsx = ttk.Label(frm, text='处理编号Excel(可选)')
    lbl_xlsx.grid(row=5, column=0, sticky='w', pady=3)
    ent_xlsx = ttk.Entry(frm, textvariable=v_xlsx, width=64)
    ent_xlsx.grid(row=5, column=1, padx=5)

    def pick_xlsx():
        p = filedialog.askopenfilename(filetypes=[('Excel 文件', '*.xlsx *.xlsm *.xls')])
        if p:
            v_xlsx.set(p)

    def gen_template():
        ids = sorted(set(scan_inputs(vars_['ref'].get(), vars_['photo'].get(), lambda s: None)[0]))
        if not ids:
            ids = ['示例编号100001']
        out_p = os.path.join(vars_['out'].get() or base, '处理编号模板.xlsx')
        try:
            write_ids_template(out_p, ids)
            v_xlsx.set(out_p)
            log(f'已生成编号模板: {out_p} (删除不需要的行即可)')
        except Exception as e:
            messagebox.showerror('错误', f'生成模板失败: {e}')

    btn_frm_x = ttk.Frame(frm)
    btn_frm_x.grid(row=5, column=2)
    ttk.Button(btn_frm_x, text='浏览...', command=pick_xlsx).pack(side='left')
    ttk.Button(btn_frm_x, text='生成模板', command=gen_template).pack(side='left', padx=(4, 0))

    opt_frm = ttk.Frame(root, padding=(10, 0))
    opt_frm.pack(fill='x')
    v_preview = tk.BooleanVar(value=True)
    v_exif = tk.BooleanVar(value=True)
    ttk.Checkbutton(opt_frm, text='生成对照预览图', variable=v_preview).pack(side='left', padx=5)
    ttk.Checkbutton(opt_frm, text='保留照片EXIF定位信息', variable=v_exif).pack(side='left', padx=5)
    lbl_ids = ttk.Label(opt_frm, text='仅处理编号(逗号分隔, 空=全部, Excel优先):')
    lbl_ids.pack(side='left', padx=(20, 3))
    v_ids = tk.StringVar(value='')
    ent_ids = ttk.Entry(opt_frm, textvariable=v_ids, width=16)
    ent_ids.pack(side='left')

    def on_mode():
        single = mode_var.get() == 'single'
        lbl_ref.configure(text='正射影像图文件' if single else '正射影像图文件夹')
        st = 'disabled' if single else 'normal'
        for wgt in (ent_xlsx, ent_ids):
            wgt.configure(state=st)
        for b in btn_frm_x.winfo_children():
            b.configure(state=st)
        if single:
            vars_['ref'].set('')
        else:
            vars_['ref'].set(os.path.join(base, '无人机正射影像图'))
    on_mode()

    prog = ttk.Progressbar(root, mode='determinate')
    prog.pack(fill='x', padx=10, pady=6)

    log_box = scrolledtext.ScrolledText(root, font=('Consolas', 9), height=22)
    log_box.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    btn_frm = ttk.Frame(root, padding=6)
    btn_frm.pack(fill='x')
    btn_run = ttk.Button(btn_frm, text='开始处理')
    btn_run.pack(side='left', padx=5)
    btn_cancel = ttk.Button(btn_frm, text='取消', state='disabled')
    btn_cancel.pack(side='left', padx=5)

    cancel_ev = threading.Event()

    def log(s):
        def _append():
            log_box.insert('end', str(s) + '\n')
            log_box.see('end')
        try:
            root.after(0, _append)
        except Exception:
            pass

    def worker():
        try:
            ids = [x.strip() for x in v_ids.get().replace('，', ',').split(',') if x.strip()]
            opts = {'make_preview': v_preview.get(), 'keep_exif': v_exif.get(),
                    'quality': 92, 'only_ids': ids, 'ids_file': v_xlsx.get().strip()}
            if mode_var.get() == 'single':
                run_single(vars_['ref'].get(), vars_['photo'].get(), vars_['out'].get(),
                           opts, log,
                           progress=lambda f: root.after(0, lambda: prog.configure(value=f * 100)),
                           cancel=cancel_ev)
            else:
                run_batch(vars_['ref'].get(), vars_['photo'].get(), vars_['out'].get(),
                          opts, log,
                          progress=lambda f: root.after(0, lambda: prog.configure(value=f * 100)),
                          cancel=cancel_ev)
            root.after(0, lambda: messagebox.showinfo('完成', '处理结束, 详情见输出文件夹的处理报告.csv'))
        except Exception as e:
            log(f'致命错误: {e}')
            log(traceback.format_exc())
        finally:
            root.after(0, lambda: (btn_run.configure(state='normal'),
                                   btn_cancel.configure(state='disabled')))

    def on_run():
        if mode_var.get() == 'single':
            if not os.path.isfile(vars_['ref'].get()):
                messagebox.showerror('错误', '请选择正射影像图文件')
                return
        elif not os.path.isdir(vars_['ref'].get()):
            messagebox.showerror('错误', '正射影像图文件夹不存在')
            return
        if not os.path.isdir(vars_['photo'].get()):
            messagebox.showerror('错误', '无人机照片文件夹不存在')
            return
        cancel_ev.clear()
        btn_run.configure(state='disabled')
        btn_cancel.configure(state='normal')
        prog.configure(value=0)
        threading.Thread(target=worker, daemon=True).start()

    btn_run.configure(command=on_run)
    btn_cancel.configure(command=lambda: (cancel_ev.set(), log('正在取消...')))
    log('批量模式: 正射图文件夹与照片文件夹任意命名, 只要编号一致即自动配对(编号在文件名或上级文件夹名中均可)。')
    log('单图模式: 选一张正射影像图文件 + 一个照片文件夹即可, 不要求编号。')
    log('可用"生成模板"生成编号 Excel, 删除不需要的行后作为"处理编号Excel"选择, 即只处理表中编号。')
    root.mainloop()


# ----------------------------- CLI / main -----------------------------
def main():
    parser = argparse.ArgumentParser(description='无人机照片红线自动匹配工具')
    parser.add_argument('--cli', action='store_true', help='命令行模式(不显示界面)')
    parser.add_argument('--single', action='store_true', help='单图模式: --ref 指向单张影像图文件')
    parser.add_argument('--ref', default='', help='正射影像图文件夹(单图模式为影像图文件)')
    parser.add_argument('--photo', default='', help='无人机照片文件夹')
    parser.add_argument('--out', default='', help='输出文件夹')
    parser.add_argument('--ids', default='', help='仅处理这些编号(逗号分隔)')
    parser.add_argument('--ids-file', default='', help='仅处理编号 Excel 表(优先于 --ids)')
    parser.add_argument('--no-preview', action='store_true', help='不生成对照预览图')
    parser.add_argument('--no-exif', action='store_true', help='不保留EXIF')
    args = parser.parse_args()

    if not args.cli:
        build_gui()
        return

    base = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, 'frozen', False) else __file__))
    ref_dir = args.ref or os.path.join(base, '无人机正射影像图')
    photo_dir = args.photo or os.path.join(base, '无人机照片')
    out_dir = args.out or os.path.join(base, '红线输出')
    ids = [x.strip() for x in args.ids.replace('，', ',').split(',') if x.strip()]
    opts = {'make_preview': not args.no_preview, 'keep_exif': not args.no_exif,
            'quality': 92, 'only_ids': ids, 'ids_file': args.ids_file}
    t0 = time.time()
    if args.single:
        run_single(ref_dir, photo_dir, out_dir, opts, print)
    else:
        run_batch(ref_dir, photo_dir, out_dir, opts, print)
    print(f'总耗时 {time.time() - t0:.0f} 秒')


if __name__ == '__main__':
    main()
