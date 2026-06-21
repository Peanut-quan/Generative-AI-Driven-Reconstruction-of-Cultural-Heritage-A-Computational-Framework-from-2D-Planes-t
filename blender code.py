# -*- coding: utf-8 -*-
bl_info = {
    "location": "3D Viewport > N 面板 > 建筑模型评估",
    "description": "一键导出建筑AI/传统建模所需数据：几何质量、拓扑、UV、材质、建筑语义、可编辑性、应用准备。",
    "category": "3D View",
}

import bpy
import bmesh
import csv
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime
from mathutils import Vector
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    StringProperty,
    BoolProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)

EPS = 1e-9


def r6(x):
    if x is None:
        return None
    try:
        if math.isnan(float(x)) or math.isinf(float(x)):
            return None
        return round(float(x), 6)
    except Exception:
        return x


def safe_div(a, b):
    try:
        if b == 0 or b is None:
            return None
        return a / b
    except Exception:
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def coefficient_of_variation(vals):
    m = mean(vals)
    s = std(vals)
    if m is None or abs(m) < EPS or s is None:
        return None
    return s / m


def ensure_dir(path):
    if not path:
        blend_path = bpy.data.filepath
        base = os.path.dirname(blend_path) if blend_path else os.path.expanduser("~")
        path = os.path.join(base, "model_metrics_exports_v4")
    os.makedirs(path, exist_ok=True)
    return path


def clean_filename(s):
    bad = '<>:"/\\|?*'
    out = ''.join('_' if c in bad else c for c in str(s))
    return out.strip() or 'untitled'


def write_csv(path, rows, headers=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        else:
            writer = csv.writer(f)
            writer.writerows(rows)


def scene_file_size_mb():
    try:
        if bpy.data.filepath and os.path.exists(bpy.data.filepath):
            return round(os.path.getsize(bpy.data.filepath) / (1024 * 1024), 3)
    except Exception:
        pass
    return None


def get_world_bbox_for_object(obj):
    if not hasattr(obj, "bound_box"):
        return []
    try:
        return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    except Exception:
        return []


def bbox_size(points):
    if not points:
        return (0.0, 0.0, 0.0)
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def triangle_area_3d(a, b, c):
    return ((b - a).cross(c - a)).length * 0.5


def triangle_aspect_ratio(a, b, c):
    e1 = (b - a).length
    e2 = (c - b).length
    e3 = (a - c).length
    longest = max(e1, e2, e3)
    area = triangle_area_3d(a, b, c)
    if area < EPS:
        return float("inf")
    # 近似长细比：最长边 / 对应高 = longest^2 / (2*area)
    return (longest * longest) / (2.0 * area)


def polygon_world_area(poly, coords, mesh):
    try:
        verts = [coords[i] for i in poly.vertices]
        if len(verts) < 3:
            return 0.0
        base = verts[0]
        total = 0.0
        for i in range(1, len(verts) - 1):
            total += triangle_area_3d(base, verts[i], verts[i + 1])
        return total
    except Exception:
        return 0.0


def uv_area_of_poly(poly, uv_layer):
    try:
        uvs = [uv_layer.data[i].uv.copy() for i in poly.loop_indices]
        if len(uvs) < 3:
            return 0.0
        base = uvs[0]
        total = 0.0
        for i in range(1, len(uvs) - 1):
            a = uvs[i] - base
            b = uvs[i + 1] - base
            total += abs(a.x * b.y - a.y * b.x) * 0.5
        return total
    except Exception:
        return 0.0


def close2(a, b, eps=1e-5):
    return (a - b).length <= eps


def edge_loop_auto_score(quad_ratio, vertex4_ratio, nonmanifold_edge_count, edge_count):
    q = quad_ratio or 0.0
    v4 = vertex4_ratio or 0.0
    nm_penalty = 0.0 if not edge_count else min(1.0, nonmanifold_edge_count / max(edge_count, 1))
    score = 10.0 * (0.65 * q + 0.35 * v4) * (1.0 - 0.6 * nm_penalty)
    return max(0.0, min(10.0, score))


def guess_arch_category(name):
    n = (name or "").lower()
    categories = [
        ("墙体", ["wall", "walls", "墙", "墙体", "facade", "立面"]),
        ("窗", ["window", "win", "窗", "玻璃窗"]),
        ("门", ["door", "门", "入口", "gate"]),
        ("屋顶", ["roof", "屋顶", "房顶", "瓦", "ridge"]),
        ("楼板/地面", ["floor", "slab", "ground", "地面", "楼板", "平台"]),
        ("柱", ["column", "pillar", "post", "柱"]),
        ("梁", ["beam", "梁"]),
        ("楼梯", ["stair", "stairs", "step", "楼梯", "台阶"]),
        ("栏杆", ["rail", "railing", "balcony", "栏杆", "扶手", "阳台"]),
        ("玻璃", ["glass", "玻璃"]),
        ("装饰/细部", ["decor", "trim", "ornament", "装饰", "线脚"]),
    ]
    for cat, keys in categories:
        if any(k in n for k in keys):
            return cat
    return "其他/未识别"


def is_named_non_default(name):
    if not name:
        return False
    n = name.lower().strip()
    default_prefixes = ("cube", "sphere", "cylinder", "cone", "plane", "mesh", "node", "object", "default")
    if n.startswith(default_prefixes):
        return False
    return True


def material_image_stats():
    image_nodes = []
    for mat in bpy.data.materials:
        if mat and mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.bl_idname == "ShaderNodeTexImage":
                    image_nodes.append(node.image)
    unique = []
    seen = set()
    packed = 0
    missing = 0
    missing_files = []
    for img in image_nodes:
        if img is None:
            missing += 1
            missing_files.append("<empty image node>")
            continue
        key = img.name
        if key not in seen:
            seen.add(key)
            unique.append(img)
        if getattr(img, "packed_file", None) is not None:
            packed += 1
        else:
            try:
                if img.filepath:
                    path = bpy.path.abspath(img.filepath)
                    if path and not os.path.exists(path):
                        missing += 1
                        missing_files.append(path)
            except Exception:
                pass
    return {
        "image_texture_node_count": len(image_nodes),
        "unique_image_count": len(unique),
        "packed_image_count": packed,
        "missing_external_image_count": missing,
        "missing_external_image_files": missing_files,
    }

# -----------------------------
# 高级检测：UV、相交、边环
# -----------------------------

def compute_uv_islands(mesh, uv_layer, eps=1e-5):
    if uv_layer is None or not mesh.polygons:
        return 0
    parent = list(range(len(mesh.polygons)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_loops = defaultdict(list)
    for poly in mesh.polygons:
        loops = list(poly.loop_indices)
        for local_i, li in enumerate(loops):
            try:
                ei = mesh.loops[li].edge_index
                nxt = loops[(local_i + 1) % len(loops)]
                edge_loops[ei].append((poly.index, li, nxt))
            except Exception:
                continue

    for ei, items in edge_loops.items():
        if len(items) != 2:
            continue
        p1, l1a, l1b = items[0]
        p2, l2a, l2b = items[1]
        try:
            a1 = uv_layer.data[l1a].uv
            b1 = uv_layer.data[l1b].uv
            a2 = uv_layer.data[l2a].uv
            b2 = uv_layer.data[l2b].uv
            # 共享边在UV上连通：正向或反向坐标匹配
            if (close2(a1, b2, eps) and close2(b1, a2, eps)) or (close2(a1, a2, eps) and close2(b1, b2, eps)):
                union(p1, p2)
        except Exception:
            continue
    return len(set(find(i) for i in range(len(mesh.polygons))))


def collect_uv_triangles(mesh, uv_layer):
    tris = []
    if uv_layer is None:
        return tris
    mesh.calc_loop_triangles()
    for t in mesh.loop_triangles:
        try:
            loop_indices = list(t.loops)
        except Exception:
            try:
                loop_indices = list(t.loop_indices)
            except Exception:
                continue
        if len(loop_indices) != 3:
            continue
        pts = [uv_layer.data[li].uv.copy() for li in loop_indices]
        area = abs((pts[1].x - pts[0].x) * (pts[2].y - pts[0].y) - (pts[1].y - pts[0].y) * (pts[2].x - pts[0].x)) * 0.5
        tris.append({"pts": pts, "area": area, "poly": getattr(t, "polygon_index", -1)})
    return tris


def orient2(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def point_in_tri_2d(p, tri, eps=1e-9):
    a, b, c = tri
    o1 = orient2(a, b, p)
    o2 = orient2(b, c, p)
    o3 = orient2(c, a, p)
    return (o1 >= -eps and o2 >= -eps and o3 >= -eps) or (o1 <= eps and o2 <= eps and o3 <= eps)


def segments_intersect_2d(a, b, c, d, eps=1e-9):
    def on_segment(p, q, r):
        return min(p.x, r.x) - eps <= q.x <= max(p.x, r.x) + eps and min(p.y, r.y) - eps <= q.y <= max(p.y, r.y) + eps
    o1 = orient2(a, b, c)
    o2 = orient2(a, b, d)
    o3 = orient2(c, d, a)
    o4 = orient2(c, d, b)
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True
    return False


def tri_overlap_2d(t1, t2):
    # 快速bbox
    xs1 = [p.x for p in t1]; ys1 = [p.y for p in t1]
    xs2 = [p.x for p in t2]; ys2 = [p.y for p in t2]
    if max(xs1) < min(xs2) or max(xs2) < min(xs1) or max(ys1) < min(ys2) or max(ys2) < min(ys1):
        return False
    edges1 = [(t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0])]
    edges2 = [(t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])]
    for e1 in edges1:
        for e2 in edges2:
            if segments_intersect_2d(e1[0], e1[1], e2[0], e2[1]):
                return True
    return point_in_tri_2d(t1[0], t2) or point_in_tri_2d(t2[0], t1)


def approximate_uv_overlap(mesh, uv_layer, max_triangles=12000, grid_size=48):
    if uv_layer is None:
        return None, None, "no_uv"
    tris = collect_uv_triangles(mesh, uv_layer)
    if len(tris) > max_triangles:
        return None, None, "skipped_too_many_uv_triangles"
    total_area = sum(t["area"] for t in tris)
    if not tris:
        return 0, 0.0, "ok"
    # 空间哈希，避免O(n^2)
    minx = min(min(p.x for p in t["pts"]) for t in tris)
    maxx = max(max(p.x for p in t["pts"]) for t in tris)
    miny = min(min(p.y for p in t["pts"]) for t in tris)
    maxy = max(max(p.y for p in t["pts"]) for t in tris)
    sx = max(maxx - minx, EPS)
    sy = max(maxy - miny, EPS)
    grid = defaultdict(list)
    for idx, t in enumerate(tris):
        xs = [p.x for p in t["pts"]]
        ys = [p.y for p in t["pts"]]
        gx0 = int((min(xs) - minx) / sx * grid_size)
        gx1 = int((max(xs) - minx) / sx * grid_size)
        gy0 = int((min(ys) - miny) / sy * grid_size)
        gy1 = int((max(ys) - miny) / sy * grid_size)
        gx0 = max(0, min(grid_size, gx0)); gx1 = max(0, min(grid_size, gx1))
        gy0 = max(0, min(grid_size, gy0)); gy1 = max(0, min(grid_size, gy1))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid[(gx, gy)].append(idx)

    checked = set()
    count = 0
    approx_area = 0.0
    for bucket in grid.values():
        n = len(bucket)
        for a_i in range(n):
            for b_i in range(a_i + 1, n):
                i, j = bucket[a_i], bucket[b_i]
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                if key in checked:
                    continue
                checked.add(key)
                # 同一多边形三角化产生的邻接片，不计入UV重叠
                if tris[i]["poly"] == tris[j]["poly"] and tris[i]["poly"] != -1:
                    continue
                if tri_overlap_2d(tris[i]["pts"], tris[j]["pts"]):
                    count += 1
                    approx_area += min(tris[i]["area"], tris[j]["area"])
    ratio = approx_area / total_area if total_area > EPS else 0.0
    return count, ratio, "近似检测完成；建议结合UV编辑器人工复核"


def estimate_closed_regular_edge_components(mesh, edge_faces, valence):
    # 近似“闭合规则边结构”：只统计由流形边且两端顶点均为4价构成的闭合组件。
    # 不是严格的Blender edge loop定义，仅用于辅助判断边流是否规则。
    valid_edges = set()
    for ei, faces in edge_faces.items():
        e = mesh.edges[ei]
        if len(faces) == 2 and valence.get(e.vertices[0], 0) == 4 and valence.get(e.vertices[1], 0) == 4:
            valid_edges.add(ei)
    if not valid_edges:
        return 0, None, None, 0.0
    v_to_edges = defaultdict(set)
    for ei in valid_edges:
        e = mesh.edges[ei]
        v_to_edges[e.vertices[0]].add(ei)
        v_to_edges[e.vertices[1]].add(ei)
    visited = set()
    loop_sizes = []
    for ei in list(valid_edges):
        if ei in visited:
            continue
        stack = [ei]
        comp_edges = set()
        comp_vertices = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp_edges.add(cur)
            e = mesh.edges[cur]
            comp_vertices.update(e.vertices)
            for v in e.vertices:
                for nei in v_to_edges[v]:
                    if nei not in visited:
                        stack.append(nei)
        # 如果组件内每个顶点在有效边子图中度数为2，则近似为闭合环/闭合环组
        if comp_edges:
            degrees = Counter()
            for ce in comp_edges:
                e = mesh.edges[ce]
                degrees[e.vertices[0]] += 1
                degrees[e.vertices[1]] += 1
            if degrees and all(d == 2 for d in degrees.values()):
                loop_sizes.append(len(comp_edges))
    return len(loop_sizes), mean(loop_sizes), std(loop_sizes), coefficient_of_variation(loop_sizes) or 0.0


def detect_intersecting_triangles(mesh, coords, max_triangles=15000):
    try:
        mesh.calc_loop_triangles()
        tris = list(mesh.loop_triangles)
        if len(tris) > max_triangles:
            return None, "skipped_too_many_triangles"
        tri_indices = [tuple(t.vertices) for t in tris]
        bvh = BVHTree.FromPolygons(coords, tri_indices, all_triangles=True)
        pairs = bvh.overlap(bvh)
        count = 0
        seen = set()
        for a, b in pairs:
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            va = set(tri_indices[a])
            vb = set(tri_indices[b])
            # 排除共享顶点/边的邻接三角形
            if va.intersection(vb):
                continue
            count += 1
        return count, "BVH近似检测完成；建议对疑似位置人工复核"
    except Exception as e:
        return None, "failed: %s" % str(e)


def bmesh_contiguous_check(mesh):
    try:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        bad = 0
        for e in bm.edges:
            try:
                if e.is_manifold and not e.is_contiguous:
                    bad += 1
            except Exception:
                pass
        bm.free()
        return bad
    except Exception:
        return None

def analyze_mesh_object(obj, props, depsgraph):
    used_evaluated = bool(props.use_evaluated_mesh)
    temp_mesh_owner = None
    try:
        if used_evaluated:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            temp_mesh_owner = eval_obj
        else:
            mesh = obj.data
    except Exception:
        mesh = obj.data
        used_evaluated = False

    try:
        mesh.calc_loop_triangles()
        world_coords = [obj.matrix_world @ v.co for v in mesh.vertices]
        vertex_count = len(mesh.vertices)
        edge_count = len(mesh.edges)
        face_count = len(mesh.polygons)
        loop_tri_count = len(mesh.loop_triangles)

        edge_faces = defaultdict(list)
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                try:
                    ei = mesh.loops[li].edge_index
                    edge_faces[ei].append(poly.index)
                except Exception:
                    pass

        tri_face_count = sum(1 for p in mesh.polygons if len(p.vertices) == 3)
        quad_face_count = sum(1 for p in mesh.polygons if len(p.vertices) == 4)
        ngon_face_count = sum(1 for p in mesh.polygons if len(p.vertices) > 4)
        tri_face_ratio = safe_div(tri_face_count, face_count) or 0.0
        quad_face_ratio = safe_div(quad_face_count, face_count) or 0.0
        ngon_face_ratio = safe_div(ngon_face_count, face_count) or 0.0

        boundary_edges = [ei for ei in range(edge_count) if len(edge_faces.get(ei, [])) == 1]
        loose_edges = [ei for ei in range(edge_count) if len(edge_faces.get(ei, [])) == 0]
        nonmanifold_edges = [ei for ei in range(edge_count) if len(edge_faces.get(ei, [])) != 2]
        nm_vertices = set()
        for ei in nonmanifold_edges:
            try:
                e = mesh.edges[ei]
                nm_vertices.update(e.vertices)
            except Exception:
                pass

        # 顶点价数
        valence = Counter()
        for e in mesh.edges:
            if len(e.vertices) >= 2:
                valence[e.vertices[0]] += 1
                valence[e.vertices[1]] += 1
        valences = [valence[i] for i in range(vertex_count)]
        valence_mean = mean(valences)
        valence4_count = sum(1 for v in valences if v == 4)
        valence34_count = sum(1 for v in valences if v in (3, 4))
        extraordinary_count = sum(1 for v in valences if v not in (3, 4))
        vertex_valence_4_ratio = safe_div(valence4_count, vertex_count) or 0.0
        vertex_valence_3_or_4_ratio = safe_div(valence34_count, vertex_count) or 0.0
        extraordinary_valence_ratio = safe_div(extraordinary_count, vertex_count) or 0.0

        # 边长、零长度边
        edge_lengths = []
        zero_length_edges = 0
        for e in mesh.edges:
            try:
                l = (world_coords[e.vertices[0]] - world_coords[e.vertices[1]]).length
                edge_lengths.append(l)
                if l <= props.zero_length_edge_epsilon:
                    zero_length_edges += 1
            except Exception:
                pass

        # 三角面质量
        zero_area_faces = 0
        thin_triangles = 0
        aspect_ratios = []
        for t in mesh.loop_triangles:
            try:
                a, b, c = [world_coords[i] for i in t.vertices]
                area = triangle_area_3d(a, b, c)
                if area <= props.zero_area_face_epsilon:
                    zero_area_faces += 1
                ar = triangle_aspect_ratio(a, b, c)
                aspect_ratios.append(ar if math.isfinite(ar) else None)
                if math.isfinite(ar) and ar >= props.thin_triangle_aspect_threshold:
                    thin_triangles += 1
            except Exception:
                pass
        aspect_ratios_clean = [a for a in aspect_ratios if a is not None]

        # 非平面面：三角面天然平面，只对四边/多边面有意义
        nonflat_faces = 0
        for p in mesh.polygons:
            if len(p.vertices) <= 3:
                continue
            try:
                verts = [world_coords[i] for i in p.vertices]
                n = (verts[1] - verts[0]).cross(verts[2] - verts[0])
                if n.length <= EPS:
                    continue
                n.normalize()
                max_dist = max(abs((v - verts[0]).dot(n)) for v in verts[3:]) if len(verts) > 3 else 0.0
                if max_dist > props.nonflat_face_tolerance:
                    nonflat_faces += 1
            except Exception:
                pass

        bad_contiguous = bmesh_contiguous_check(mesh)

        # 顶点近似重合
        close_pairs = 0
        if vertex_count > 0 and props.close_vertex_threshold > 0:
            kd = KDTree(vertex_count)
            for i, co in enumerate(world_coords):
                kd.insert(co, i)
            kd.balance()
            for i, co in enumerate(world_coords):
                for _, j, dist in kd.find_range(co, props.close_vertex_threshold):
                    if j > i:
                        close_pairs += 1
                        if close_pairs > props.max_close_vertex_pairs_report:
                            break
                if close_pairs > props.max_close_vertex_pairs_report:
                    break

        # 相交面近似检测
        if props.enable_self_intersection_check:
            intersecting_pairs, intersect_note = detect_intersecting_triangles(mesh, world_coords, props.max_intersection_triangles)
        else:
            intersecting_pairs, intersect_note = None, "disabled"

        # UV 统计
        uv_layer = mesh.uv_layers.active if mesh.uv_layers else None
        has_uv = uv_layer is not None
        uv_layer_count = len(mesh.uv_layers)
        uv_island_count = 0
        uv_degenerate_faces = 0
        uv_outside_loops = 0
        uv_total_area = 0.0
        uv_texel_density_values = []
        uv_overlap_pairs = None
        uv_overlap_area_ratio = None
        uv_overlap_note = "no_uv"
        if has_uv:
            uv_island_count = compute_uv_islands(mesh, uv_layer, props.uv_coord_match_epsilon)
            for p in mesh.polygons:
                ua = uv_area_of_poly(p, uv_layer)
                uv_total_area += ua
                if ua <= props.uv_degenerate_face_epsilon:
                    uv_degenerate_faces += 1
                for li in p.loop_indices:
                    try:
                        uv = uv_layer.data[li].uv
                        if uv.x < 0 or uv.x > 1 or uv.y < 0 or uv.y > 1:
                            uv_outside_loops += 1
                    except Exception:
                        pass
                ga = polygon_world_area(p, world_coords, mesh)
                if ga > EPS and ua > EPS:
                    uv_texel_density_values.append(math.sqrt(ua) / math.sqrt(ga))
            if props.enable_uv_overlap:
                uv_overlap_pairs, uv_overlap_area_ratio, uv_overlap_note = approximate_uv_overlap(
                    mesh, uv_layer, props.max_uv_overlap_triangles, props.uv_overlap_grid_size
                )

        uv_td_mean = mean(uv_texel_density_values)
        uv_td_std = std(uv_texel_density_values)
        uv_td_cv = coefficient_of_variation(uv_texel_density_values)

        # 材质统计
        used_material_indices = set()
        faces_without_material = 0
        for p in mesh.polygons:
            mi = p.material_index
            if mi < 0 or mi >= len(obj.material_slots) or obj.material_slots[mi].material is None:
                faces_without_material += 1
            else:
                used_material_indices.add(mi)

        # bbox
        bbox_pts = get_world_bbox_for_object(obj)
        sx, sy, sz = bbox_size(bbox_pts)
        try:
            center = sum(bbox_pts, Vector((0, 0, 0))) / len(bbox_pts)
            origin_to_center = (obj.matrix_world.translation - center).length
        except Exception:
            origin_to_center = None

        # 边环近似
        closed_loop_count, loop_avg, loop_std, loop_cv = estimate_closed_regular_edge_components(mesh, edge_faces, valence)
        edge_loop_score = edge_loop_auto_score(quad_face_ratio, vertex_valence_4_ratio, len(nonmanifold_edges), edge_count)

        category = guess_arch_category(obj.name)
        data = {
            "object_name": obj.name,
            "mesh_name": getattr(obj.data, "name", ""),
            "category_guess_cn": category,
            "is_arch_category_recognized": category != "其他/未识别",
            "is_named_non_default": is_named_non_default(obj.name),
            "is_selected": bool(obj.select_get()),
            "is_hidden_viewport": bool(obj.hide_get()),
            "used_evaluated_mesh": used_evaluated,
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "face_count": face_count,
            "loop_triangle_count": loop_tri_count,
            "tri_face_count": tri_face_count,
            "quad_face_count": quad_face_count,
            "ngon_face_count": ngon_face_count,
            "tri_face_ratio": r6(tri_face_ratio),
            "quad_face_ratio": r6(quad_face_ratio),
            "ngon_face_ratio": r6(ngon_face_ratio),
            "vertex_valence_mean": r6(valence_mean),
            "vertex_valence_4_count": valence4_count,
            "vertex_valence_4_ratio": r6(vertex_valence_4_ratio),
            "vertex_valence_3_or_4_ratio": r6(vertex_valence_3_or_4_ratio),
            "extraordinary_valence_count": extraordinary_count,
            "extraordinary_valence_ratio": r6(extraordinary_valence_ratio),
            "estimated_closed_regular_edge_loop_count": closed_loop_count,
            "estimated_edge_loop_size_mean": r6(loop_avg),
            "estimated_edge_loop_size_std": r6(loop_std),
            "estimated_edge_loop_size_cv": r6(loop_cv),
            "edge_loop_auto_score_0_10": r6(edge_loop_score),
            "nonmanifold_edge_count": len(nonmanifold_edges),
            "boundary_edge_count": len(boundary_edges),
            "loose_edge_count": len(loose_edges),
            "nonmanifold_vertex_count": len(nm_vertices),
            "bad_contiguous_edge_count": bad_contiguous,
            "intersecting_triangle_pair_count": intersecting_pairs,
            "intersecting_triangle_note": intersect_note,
            "close_vertex_pair_count": close_pairs,
            "close_vertex_pair_note": "capped" if close_pairs > props.max_close_vertex_pairs_report else "ok",
            "zero_area_face_count": zero_area_faces,
            "zero_length_edge_count": zero_length_edges,
            "nonflat_face_count": nonflat_faces,
            "thin_triangle_count": thin_triangles,
            "thin_triangle_ratio": r6(safe_div(thin_triangles, loop_tri_count) or 0.0),
            "triangle_aspect_ratio_mean": r6(mean(aspect_ratios_clean)),
            "triangle_aspect_ratio_max": r6(max(aspect_ratios_clean) if aspect_ratios_clean else None),
            "edge_length_mean": r6(mean(edge_lengths)),
            "edge_length_std": r6(std(edge_lengths)),
            "edge_length_cv": r6(coefficient_of_variation(edge_lengths)),
            "edge_length_min": r6(min(edge_lengths) if edge_lengths else None),
            "edge_length_max": r6(max(edge_lengths) if edge_lengths else None),
            "material_slot_count": len(obj.material_slots),
            "used_material_count": len(used_material_indices),
            "faces_without_material_count": faces_without_material,
            "faces_without_material_ratio": r6(safe_div(faces_without_material, face_count) or 0.0),
            "bbox_size_x": r6(sx),
            "bbox_size_y": r6(sy),
            "bbox_size_z": r6(sz),
            "origin_to_bbox_center_distance": r6(origin_to_center),
            "scale_x": r6(obj.scale.x),
            "scale_y": r6(obj.scale.y),
            "scale_z": r6(obj.scale.z),
            "scale_applied_near_1": abs(obj.scale.x - 1.0) < 1e-4 and abs(obj.scale.y - 1.0) < 1e-4 and abs(obj.scale.z - 1.0) < 1e-4,
            "has_negative_scale": obj.scale.x < 0 or obj.scale.y < 0 or obj.scale.z < 0,
            "collection_names": [c.name for c in obj.users_collection],
            "has_uv": has_uv,
            "uv_layer_count": uv_layer_count,
            "active_uv_name": uv_layer.name if has_uv else "",
            "uv_island_count": uv_island_count,
            "uv_degenerate_face_count": uv_degenerate_faces,
            "uv_outside_0_1_loop_count": uv_outside_loops,
            "uv_total_area": r6(uv_total_area),
            "uv_relative_texel_density_mean": r6(uv_td_mean),
            "uv_relative_texel_density_std": r6(uv_td_std),
            "uv_relative_texel_density_cv": r6(uv_td_cv),
            "uv_overlap_pairs": uv_overlap_pairs,
            "uv_overlap_area_ratio": r6(uv_overlap_area_ratio),
            "uv_overlap_note": uv_overlap_note,
        }
        return data
    finally:
        if temp_mesh_owner is not None:
            try:
                temp_mesh_owner.to_mesh_clear()
            except Exception:
                pass

# -----------------------------
# 汇总与输出
# -----------------------------

def sum_field(objects, field):
    total = 0
    has = False
    for o in objects:
        v = o.get(field)
        if isinstance(v, (int, float)) and v is not None:
            total += v
            has = True
    return total if has else None


def mean_field(objects, field):
    vals = [o.get(field) for o in objects if isinstance(o.get(field), (int, float))]
    return r6(mean(vals)) if vals else None


def build_summary(objects, props, elapsed):
    total_faces = sum_field(objects, "face_count") or 0
    tri = sum_field(objects, "tri_face_count") or 0
    quad = sum_field(objects, "quad_face_count") or 0
    ngon = sum_field(objects, "ngon_face_count") or 0
    analyzed_count = len(objects)
    named_count = sum(1 for o in objects if o.get("is_named_non_default"))
    arch_named_count = sum(1 for o in objects if o.get("is_arch_category_recognized"))
    bbox_pts = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and (not props.selected_only or obj.select_get()):
            bbox_pts.extend(get_world_bbox_for_object(obj))
    sx, sy, sz = bbox_size(bbox_pts)

    manual_total = (
        props.initial_modeling_minutes +
        props.regeneration_or_revision_minutes +
        props.cleanup_minutes +
        props.topology_fix_minutes +
        props.uv_minutes +
        props.material_minutes +
        props.export_test_minutes
    )
    edit_vals = [
        props.move_window_minutes,
        props.delete_door_fill_wall_minutes,
        props.change_wall_height_minutes,
        props.replace_roof_material_minutes,
    ]
    edit_nonzero = [v for v in edit_vals if v > 0]

    img_stats = material_image_stats()
    armature_count = sum(1 for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    armature_mod_total = 0
    mesh_with_armature_mod = 0
    vertex_group_count = 0
    mesh_with_vertex_groups = 0
    mesh_with_shape_keys = 0
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    for o in mesh_objs:
        amods = [m for m in o.modifiers if m.type == 'ARMATURE']
        armature_mod_total += len(amods)
        if amods:
            mesh_with_armature_mod += 1
        vertex_group_count += len(o.vertex_groups)
        if len(o.vertex_groups) > 0:
            mesh_with_vertex_groups += 1
        if getattr(o.data, "shape_keys", None) is not None:
            mesh_with_shape_keys += 1

    rig_success_rate = safe_div(props.rigging_success_count, props.rigging_test_total) if props.rigging_test_total > 0 else None

    summary = {
        "project_tag": props.project_tag,
        "blend_file": bpy.data.filepath,
        "export_time": datetime.now().isoformat(timespec="seconds"),
        "blender_version": bpy.app.version_string,
        "python_version": platform.python_version(),
        "system": platform.platform(),
        "selected_only": bool(props.selected_only),
        "use_evaluated_mesh": bool(props.use_evaluated_mesh),
        "file_size_mb": scene_file_size_mb(),
        "object_count_scene_total": len(bpy.context.scene.objects),
        "mesh_object_count_scene_total": len(mesh_objs),
        "object_count_analyzed": analyzed_count,
        "material_count_scene_total": len(bpy.data.materials),
        "collection_count_scene_total": len(bpy.data.collections),
        "vertex_count_total": sum_field(objects, "vertex_count") or 0,
        "edge_count_total": sum_field(objects, "edge_count") or 0,
        "face_count_total": total_faces,
        "loop_triangle_count_total": sum_field(objects, "loop_triangle_count") or 0,
        "tri_face_count_total": tri,
        "quad_face_count_total": quad,
        "ngon_face_count_total": ngon,
        "tri_face_ratio_total": r6(safe_div(tri, total_faces) or 0.0),
        "quad_face_ratio_total": r6(safe_div(quad, total_faces) or 0.0),
        "ngon_face_ratio_total": r6(safe_div(ngon, total_faces) or 0.0),
        "nonmanifold_edge_count_total": sum_field(objects, "nonmanifold_edge_count") or 0,
        "boundary_edge_count_total": sum_field(objects, "boundary_edge_count") or 0,
        "loose_edge_count_total": sum_field(objects, "loose_edge_count") or 0,
        "nonmanifold_vertex_count_total": sum_field(objects, "nonmanifold_vertex_count") or 0,
        "bad_contiguous_edge_count_total": sum_field(objects, "bad_contiguous_edge_count"),
        "intersecting_triangle_pair_count_total": sum_field(objects, "intersecting_triangle_pair_count"),
        "zero_area_face_count_total": sum_field(objects, "zero_area_face_count") or 0,
        "zero_length_edge_count_total": sum_field(objects, "zero_length_edge_count") or 0,
        "nonflat_face_count_total": sum_field(objects, "nonflat_face_count") or 0,
        "thin_triangle_count_total": sum_field(objects, "thin_triangle_count") or 0,
        "thin_triangle_ratio_total": r6(safe_div(sum_field(objects, "thin_triangle_count") or 0, sum_field(objects, "loop_triangle_count") or 0) or 0.0),
        "close_vertex_pair_count_total": sum_field(objects, "close_vertex_pair_count") or 0,
        "uv_object_count": sum(1 for o in objects if o.get("has_uv")),
        "uv_object_ratio": r6(safe_div(sum(1 for o in objects if o.get("has_uv")), analyzed_count) or 0.0),
        "uv_island_count_total": sum_field(objects, "uv_island_count") or 0,
        "uv_overlap_pairs_total": sum_field(objects, "uv_overlap_pairs"),
        "uv_overlap_area_ratio_mean": mean_field(objects, "uv_overlap_area_ratio"),
        "uv_degenerate_face_count_total": sum_field(objects, "uv_degenerate_face_count") or 0,
        "uv_outside_0_1_loop_count_total": sum_field(objects, "uv_outside_0_1_loop_count") or 0,
        "uv_relative_texel_density_cv_mean": mean_field(objects, "uv_relative_texel_density_cv"),
        "edge_length_std_mean": mean_field(objects, "edge_length_std"),
        "edge_length_cv_mean": mean_field(objects, "edge_length_cv"),
        "vertex_valence_4_ratio_mean": mean_field(objects, "vertex_valence_4_ratio"),
        "estimated_closed_regular_edge_loop_count_total": sum_field(objects, "estimated_closed_regular_edge_loop_count") or 0,
        "estimated_edge_loop_size_cv_mean": mean_field(objects, "estimated_edge_loop_size_cv"),
        "edge_loop_auto_score_mean_0_10": mean_field(objects, "edge_loop_auto_score_0_10"),
        "scene_bbox_size_x": r6(sx),
        "scene_bbox_size_y": r6(sy),
        "scene_bbox_size_z": r6(sz),
        "named_mesh_object_count": named_count,
        "named_mesh_object_ratio": r6(safe_div(named_count, analyzed_count) or 0.0),
        "architectural_category_named_count": arch_named_count,
        "architectural_category_named_ratio": r6(safe_div(arch_named_count, analyzed_count) or 0.0),
        "one_piece_mesh_warning": analyzed_count == 1 and arch_named_count == 0,
        "unapplied_scale_object_count": sum(1 for o in objects if not o.get("scale_applied_near_1")),
        "negative_scale_object_count": sum(1 for o in objects if o.get("has_negative_scale")),
        "material_slot_count_total": sum_field(objects, "material_slot_count") or 0,
        "used_material_count_total": sum_field(objects, "used_material_count") or 0,
        "faces_without_material_count_total": sum_field(objects, "faces_without_material_count") or 0,
        "faces_without_material_ratio_total": r6(safe_div(sum_field(objects, "faces_without_material_count") or 0, total_faces) or 0.0),
        **img_stats,
        # 建筑准确性：参考值 + 自动计算
        "reference_size_x": props.reference_size_x,
        "reference_size_y": props.reference_size_y,
        "reference_size_z": props.reference_size_z,
        "size_error_x_ratio": r6(abs(sx - props.reference_size_x) / props.reference_size_x) if props.reference_size_x > 0 else None,
        "size_error_y_ratio": r6(abs(sy - props.reference_size_y) / props.reference_size_y) if props.reference_size_y > 0 else None,
        "size_error_z_ratio": r6(abs(sz - props.reference_size_z) / props.reference_size_z) if props.reference_size_z > 0 else None,
        "size_error_mean_ratio": None,
        "manual_reference_door_count": props.reference_door_count,
        "manual_detected_correct_door_count": props.detected_correct_door_count,
        "door_count_accuracy": r6(safe_div(props.detected_correct_door_count, props.reference_door_count)) if props.reference_door_count > 0 else None,
        "manual_reference_window_count": props.reference_window_count,
        "manual_detected_correct_window_count": props.detected_correct_window_count,
        "window_count_accuracy": r6(safe_div(props.detected_correct_window_count, props.reference_window_count)) if props.reference_window_count > 0 else None,
        "manual_reference_opening_count": props.reference_opening_count,
        "manual_correct_opening_count": props.correct_opening_count,
        "opening_integrity_ratio": r6(safe_div(props.correct_opening_count, props.reference_opening_count)) if props.reference_opening_count > 0 else None,
        "manual_component_intersection_count": props.component_intersection_count,
        "manual_component_gap_count": props.component_gap_count,
        "manual_z_fighting_count": props.z_fighting_count,
        # 五标准人工项
        "manual_initial_modeling_minutes": props.initial_modeling_minutes,
        "manual_regeneration_or_revision_minutes": props.regeneration_or_revision_minutes,
        "manual_cleanup_minutes": props.cleanup_minutes,
        "manual_topology_fix_minutes": props.topology_fix_minutes,
        "manual_uv_minutes": props.uv_minutes,
        "manual_material_minutes": props.material_minutes,
        "manual_export_test_minutes": props.export_test_minutes,
        "manual_total_minutes": r6(manual_total),
        "manual_move_window_minutes": props.move_window_minutes,
        "manual_delete_door_fill_wall_minutes": props.delete_door_fill_wall_minutes,
        "manual_change_wall_height_minutes": props.change_wall_height_minutes,
        "manual_replace_roof_material_minutes": props.replace_roof_material_minutes,
        "manual_edit_task_average_minutes": r6(mean(edit_nonzero)) if edit_nonzero else None,
        "manual_edge_loop_placement_score_1_10": props.manual_edge_loop_placement_score,
        "manual_uv_overlap_percent": props.manual_uv_overlap_percent,
        "manual_uv_stretching_score_1_10": props.manual_uv_stretching_score,
        "manual_texture_application_ready_score_1_10": props.manual_texture_application_ready_score,
        "manual_texture_space_utilization_score_1_10": props.manual_texture_space_utilization_score,
        "manual_seam_placement_score_1_10": props.manual_seam_placement_score,
        # 应用准备
        "armature_object_count": armature_count,
        "armature_modifier_total": armature_mod_total,
        "mesh_with_armature_modifier_count": mesh_with_armature_mod,
        "mesh_with_armature_modifier_ratio": r6(safe_div(mesh_with_armature_mod, len(mesh_objs)) or 0.0),
        "vertex_group_count_total": vertex_group_count,
        "mesh_with_vertex_groups_count": mesh_with_vertex_groups,
        "mesh_with_vertex_groups_ratio": r6(safe_div(mesh_with_vertex_groups, len(mesh_objs)) or 0.0),
        "mesh_with_shape_keys_count": mesh_with_shape_keys,
        "basic_rigging_setup_detected": bool(armature_count or armature_mod_total or vertex_group_count),
        "manual_rigging_test_total": props.rigging_test_total,
        "manual_rigging_success_count": props.rigging_success_count if props.rigging_test_total > 0 else None,
        "manual_rigging_failure_count": props.rigging_test_total - props.rigging_success_count if props.rigging_test_total > 0 else None,
        "manual_rigging_success_rate": r6(rig_success_rate) if rig_success_rate is not None else None,
        "manual_deformation_error_count": props.deformation_error_count,
        "manual_deformation_error_note": props.deformation_error_note,
        "application_rigging_auto_score_0_10": r6(10.0 * (mesh_with_vertex_groups / max(len(mesh_objs), 1)) if armature_count else 0.0),
        "elapsed_seconds": r6(elapsed),
    }
    errs = [summary.get("size_error_x_ratio"), summary.get("size_error_y_ratio"), summary.get("size_error_z_ratio")]
    errs = [e for e in errs if e is not None]
    summary["size_error_mean_ratio"] = r6(mean(errs)) if errs else None
    return summary


def metric_explanations_cn():
    return [
        {"category": "建模效率", "metric": "初始生成/建模时间", "explanation": "AI生成或传统手工初始建模所需时间，单位分钟。", "source": "人工填写"},
        {"category": "建模效率", "metric": "清理/拓扑修复时间", "explanation": "删除破面、修复拓扑、合并点、整理对象、处理UV等耗时。", "source": "人工填写"},
        {"category": "基础几何质量", "metric": "非流形边、边界边、相交面、零面积面、零长度边、非平面面", "explanation": "参考CAD/3D模型可用性检查，用于判断模型是否存在会影响布尔、渲染、导入和后续编辑的结构错误。", "source": "自动统计/近似检测"},
        {"category": "拓扑质量", "metric": "三角面比例、四边面比例、顶点4价比例、边环估计", "explanation": "用于判断网格是否具有适合编辑、变形、构件修改的边流结构。", "source": "自动统计+人工评分"},
        {"category": "UV布局", "metric": "UV岛、UV重叠、UV退化、0-1外UV、Texel Density CV", "explanation": "用于评价UV是否适合贴图、烘焙、材质替换和建筑可视化。", "source": "自动统计+人工复核"},
        {"category": "UV布局", "metric": "接缝位置、纹理空间利用率、UV拉伸", "explanation": "接缝是否在显眼区域、UV空间是否浪费、贴图是否拉伸。", "source": "人工评分"},
        {"category": "建筑语义", "metric": "对象数、命名比例、建筑构件识别比例", "explanation": "判断模型是否按墙、窗、门、屋顶等构件组织。", "source": "自动统计+命名规则"},
        {"category": "建筑准确性", "metric": "尺寸误差、门窗数量准确率、开洞完整率、穿插/缝隙", "explanation": "尺寸可通过参考值自动计算，门窗和洞口需结合参考图或人工标注。", "source": "人工填写+自动计算"},
        {"category": "可编辑性", "metric": "编辑任务耗时", "explanation": "移动窗户、删除门补墙、修改墙高、替换材质等任务耗时，用于量化后期修改难度。", "source": "人工测试"},
        {"category": "应用准备", "metric": "文件大小、贴图缺失、缩放状态、骨架/顶点组、装配测试", "explanation": "用于判断模型是否适合导出、引擎导入、漫游、绑定或后续流程。", "source": "自动统计+人工测试"},
    ]


def build_paper_rows(summary):
    def val(k):
        return summary.get(k, "")
    rows = [
        {"评价维度": "基础规模", "指标": "文件大小 / MB", "数值": val("file_size_mb"), "解释": "文件体积反映资产管理与传输成本。"},
        {"评价维度": "基础规模", "指标": "Mesh对象数", "数值": val("object_count_analyzed"), "解释": "对象数越多通常越利于局部选择，但也可能增加管理复杂度。"},
        {"评价维度": "拓扑质量", "指标": "顶点数", "数值": val("vertex_count_total"), "解释": "衡量模型几何复杂度。"},
        {"评价维度": "拓扑质量", "指标": "面数", "数值": val("face_count_total"), "解释": "衡量模型网格规模和渲染负担。"},
        {"评价维度": "拓扑质量", "指标": "三角面比例", "数值": val("tri_face_ratio_total"), "解释": "过高通常说明后期边环编辑能力较弱。"},
        {"评价维度": "拓扑质量", "指标": "四边面比例", "数值": val("quad_face_ratio_total"), "解释": "建筑硬表面模型中较高四边面比例通常更利于编辑。"},
        {"评价维度": "基础几何质量", "指标": "非流形边", "数值": val("nonmanifold_edge_count_total"), "解释": "影响闭合性、布尔运算和后续CAD/渲染流程。"},
        {"评价维度": "基础几何质量", "指标": "边界边/开放边", "数值": val("boundary_edge_count_total"), "解释": "反映是否存在开放结构或未闭合区域。"},
        {"评价维度": "基础几何质量", "指标": "相交三角面对", "数值": val("intersecting_triangle_pair_count_total"), "解释": "反映自相交风险；大模型可设置上限或人工复核。"},
        {"评价维度": "基础几何质量", "指标": "零面积面", "数值": val("zero_area_face_count_total"), "解释": "退化几何会影响渲染、布尔、仿真和导入。"},
        {"评价维度": "基础几何质量", "指标": "零长度边", "数值": val("zero_length_edge_count_total"), "解释": "无效边会影响模型稳定性。"},
        {"评价维度": "拓扑质量", "指标": "细长三角形比例", "数值": val("thin_triangle_ratio_total"), "解释": "比例越高，局部网格越不稳定。"},
        {"评价维度": "拓扑质量", "指标": "顶点4价比例", "数值": val("vertex_valence_4_ratio_mean"), "解释": "可辅助判断网格是否接近规则四边面结构。"},
        {"评价维度": "拓扑质量", "指标": "闭合规则边环估计数", "数值": val("estimated_closed_regular_edge_loop_count_total"), "解释": "用于近似判断边流/边环是否有规律。"},
        {"评价维度": "UV布局", "指标": "UV岛数量", "数值": val("uv_island_count_total"), "解释": "岛数量过多可能说明UV碎片化。"},
        {"评价维度": "UV布局", "指标": "UV重叠对", "数值": val("uv_overlap_pairs_total"), "解释": "重叠会影响贴图烘焙和材质显示。"},
        {"评价维度": "UV布局", "指标": "UV退化面", "数值": val("uv_degenerate_face_count_total"), "解释": "退化UV会导致贴图异常。"},
        {"评价维度": "UV布局", "指标": "UV超出0-1 Loop", "数值": val("uv_outside_0_1_loop_count_total"), "解释": "用于判断UV是否在标准贴图空间内。"},
        {"评价维度": "UV布局", "指标": "Texel Density CV", "数值": val("uv_relative_texel_density_cv_mean"), "解释": "越高说明贴图清晰度分布越不均匀。"},
        {"评价维度": "材质贴图", "指标": "实际使用材质数", "数值": val("used_material_count_total"), "解释": "反映材质管理复杂度。"},
        {"评价维度": "材质贴图", "指标": "缺失贴图数量", "数值": val("missing_external_image_count"), "解释": "判断迁移和导出后的材质完整性。"},
        {"评价维度": "可编辑性", "指标": "对象命名比例", "数值": val("named_mesh_object_ratio"), "解释": "命名越规范越便于后期管理。"},
        {"评价维度": "可编辑性", "指标": "建筑构件识别比例", "数值": val("architectural_category_named_ratio"), "解释": "判断是否按墙、窗、门、屋顶等建筑语义组织。"},
        {"评价维度": "可编辑性", "指标": "一体化网格警告", "数值": val("one_piece_mesh_warning"), "解释": "一体化网格通常会降低构件级编辑能力。"},
        {"评价维度": "应用准备", "指标": "未应用缩放对象数", "数值": val("unapplied_scale_object_count"), "解释": "缩放未应用可能影响导出、测量和绑定。"},
        {"评价维度": "应用准备", "指标": "基础装配/绑定设置", "数值": val("basic_rigging_setup_detected"), "解释": "建筑模型可转化为装配、交互或变形测试准备度。"},
    ]
    return rows


def build_framework_rows():
    return [
        {"评价模块": "基础几何质量", "关键指标": "非流形边、错误连续边、相交面、零面积面、零长度边、非平面面", "适用意义": "判断模型是否存在会影响布尔、CAD、渲染和导入的结构错误", "数据来源": "自动统计/近似检测"},
        {"评价模块": "拓扑结构", "关键指标": "三角面比例、四边面比例、顶点4价比例、闭合边环估计、边长CV", "适用意义": "判断模型是否适合建筑构件修改、重拓扑、变形或装配", "数据来源": "自动统计+人工评分"},
        {"评价模块": "UV布局", "关键指标": "UV岛、UV重叠、UV退化、0-1外UV、Texel Density CV、接缝位置、纹理空间利用率", "适用意义": "判断模型是否适合贴图、烘焙、建筑可视化和数字孪生展示", "数据来源": "自动统计+UV截图复核"},
        {"评价模块": "建筑语义结构", "关键指标": "Mesh对象数、对象命名比例、建筑构件识别比例、一体化网格", "适用意义": "判断墙、窗、门、屋顶等构件是否清晰可编辑", "数据来源": "自动统计+命名规则"},
        {"评价模块": "建筑准确性", "关键指标": "尺寸误差、门窗数量准确率、开洞完整率、构件穿插/缝隙", "适用意义": "判断模型是否真实表达建筑结构，而不只是视觉相似", "数据来源": "参考图/图纸/人工填写"},
        {"评价模块": "可编辑性", "指标": "移动窗户、删除门补墙、修改墙高、替换屋顶材质、清理拓扑时间", "适用意义": "用时间量化后期修改成本", "数据来源": "人工操作记录"},
        {"评价模块": "应用准备", "关键指标": "文件大小、贴图缺失、缩放状态、导出测试、引擎导入、基础装配/绑定", "适用意义": "判断模型是否能进入渲染、漫游、AR/VR或CAD/CAM/CAE相关流程", "数据来源": "自动统计+人工测试"},
    ]

# -----------------------------
# UI属性
# -----------------------------

class ARCHMETRICS_V4_Properties(PropertyGroup):
    project_tag: StringProperty(name="项目标签", default="AI", description="建议填写 AI 或 Traditional")
    output_dir: StringProperty(name="输出文件夹", subtype='DIR_PATH', default="")
    selected_only: BoolProperty(name="只统计已选择对象", default=False)
    use_evaluated_mesh: BoolProperty(name="统计应用修改器后的网格", default=True)

    # 高级自动检测开关
    enable_uv_overlap: BoolProperty(name="检测UV重叠（近似）", default=True)
    max_uv_overlap_triangles: IntProperty(name="UV重叠检测三角面上限", default=12000, min=100, description="模型太大时建议先用默认值；要完整检测可提高，但会变慢")
    uv_overlap_grid_size: IntProperty(name="UV重叠空间哈希网格", default=48, min=8, max=256)
    enable_self_intersection_check: BoolProperty(name="检测自相交面（近似）", default=False, description="大模型会较慢，建议需要时开启")
    max_intersection_triangles: IntProperty(name="相交面检测三角面上限", default=15000, min=100)

    # 阈值
    close_vertex_threshold: FloatProperty(name="重合点阈值", default=0.0001, min=0.0, precision=6)
    max_close_vertex_pairs_report: IntProperty(name="重合点对报告上限", default=200000, min=1000)
    thin_triangle_aspect_threshold: FloatProperty(name="细长三角形阈值", default=10.0, min=1.0, description="最长边/对应高大于该值视为细长三角形")
    zero_area_face_epsilon: FloatProperty(name="零面积面阈值", default=1e-10, min=0.0, precision=10)
    zero_length_edge_epsilon: FloatProperty(name="零长度边阈值", default=1e-8, min=0.0, precision=10)
    nonflat_face_tolerance: FloatProperty(name="非平面面容差", default=1e-5, min=0.0, precision=8)
    uv_degenerate_face_epsilon: FloatProperty(name="退化UV面阈值", default=1e-10, min=0.0, precision=10)
    uv_coord_match_epsilon: FloatProperty(name="UV岛连接容差", default=1e-5, min=0.0, precision=8)

    # 构建时间
    initial_modeling_minutes: FloatProperty(name="初始生成/建模时间 min", default=0.0, min=0.0)
    regeneration_or_revision_minutes: FloatProperty(name="重新生成/修改迭代时间 min", default=0.0, min=0.0)
    cleanup_minutes: FloatProperty(name="清理修复时间 min", default=0.0, min=0.0)
    topology_fix_minutes: FloatProperty(name="拓扑问题修复时间 min", default=0.0, min=0.0)
    uv_minutes: FloatProperty(name="UV处理时间 min", default=0.0, min=0.0)
    material_minutes: FloatProperty(name="材质处理时间 min", default=0.0, min=0.0)
    export_test_minutes: FloatProperty(name="导出/应用测试时间 min", default=0.0, min=0.0)

    # 建筑准确性
    reference_size_x: FloatProperty(name="参考尺寸X", default=0.0, min=0.0)
    reference_size_y: FloatProperty(name="参考尺寸Y", default=0.0, min=0.0)
    reference_size_z: FloatProperty(name="参考尺寸Z", default=0.0, min=0.0)
    reference_door_count: IntProperty(name="参考门数量", default=0, min=0)
    detected_correct_door_count: IntProperty(name="正确门数量", default=0, min=0)
    reference_window_count: IntProperty(name="参考窗数量", default=0, min=0)
    detected_correct_window_count: IntProperty(name="正确窗数量", default=0, min=0)
    reference_opening_count: IntProperty(name="参考开洞数量", default=0, min=0)
    correct_opening_count: IntProperty(name="正确开洞数量", default=0, min=0)
    component_intersection_count: IntProperty(name="构件穿插数量", default=0, min=0)
    component_gap_count: IntProperty(name="构件缝隙数量", default=0, min=0)
    z_fighting_count: IntProperty(name="Z-fighting区域数量", default=0, min=0)

    # 可编辑性任务
    move_window_minutes: FloatProperty(name="移动窗户任务时间 min", default=0.0, min=0.0)
    delete_door_fill_wall_minutes: FloatProperty(name="删除门并补墙时间 min", default=0.0, min=0.0)
    change_wall_height_minutes: FloatProperty(name="修改墙高时间 min", default=0.0, min=0.0)
    replace_roof_material_minutes: FloatProperty(name="替换屋顶材质时间 min", default=0.0, min=0.0)

    # 拓扑/UV人工评分
    manual_edge_loop_placement_score: FloatProperty(name="边环放置评分 1-10", default=0.0, min=0.0, max=10.0)
    manual_uv_overlap_percent: FloatProperty(name="UV重叠比例 %", default=0.0, min=0.0, max=100.0)
    manual_uv_stretching_score: FloatProperty(name="UV拉伸评分 1-10", default=0.0, min=0.0, max=10.0)
    manual_texture_application_ready_score: FloatProperty(name="纹理应用准备评分 1-10", default=0.0, min=0.0, max=10.0)
    manual_texture_space_utilization_score: FloatProperty(name="纹理空间利用率评分 1-10", default=0.0, min=0.0, max=10.0)
    manual_seam_placement_score: FloatProperty(name="UV接缝位置评分 1-10", default=0.0, min=0.0, max=10.0)

    # 装配/应用测试
    rigging_test_total: IntProperty(name="装配/绑定测试次数", default=0, min=0)
    rigging_success_count: IntProperty(name="装配/绑定成功次数", default=0, min=0)
    deformation_error_count: IntProperty(name="变形/交互错误数量", default=0, min=0)
    deformation_error_note: StringProperty(name="变形/交互错误说明", default="")


# -----------------------------
# 导出操作
# -----------------------------

class ARCHMETRICS_V4_OT_Export(Operator):
    bl_idname = "archmetrics_v4.export_metrics_cn"
    bl_label = "一键导出模型评估数据 v4"
    bl_description = "导出AI/传统建筑建模所需的JSON、中文CSV"
    bl_options = {'REGISTER'}

    def execute(self, context):
        start = time.time()
        props = context.scene.archmetrics_v4_props
        depsgraph = context.evaluated_depsgraph_get()
        mesh_objects = [o for o in context.scene.objects if o.type == 'MESH']
        if props.selected_only:
            mesh_objects = [o for o in mesh_objects if o.select_get()]
        if not mesh_objects:
            self.report({'ERROR'}, "没有可分析的Mesh对象")
            return {'CANCELLED'}

        objects_data = []
        for obj in mesh_objects:
            try:
                objects_data.append(analyze_mesh_object(obj, props, depsgraph))
            except Exception as e:
                objects_data.append({"object_name": obj.name, "error": str(e)})

        elapsed = time.time() - start
        summary = build_summary(objects_data, props, elapsed)
        component_counts = Counter(o.get("category_guess_cn", "其他/未识别") for o in objects_data)
        component_breakdown = [
            {"category_cn": cat, "object_count": cnt, "ratio": r6(safe_div(cnt, len(objects_data)) or 0.0)}
            for cat, cnt in component_counts.most_common()
        ]
        report = {
            "summary": summary,
            "objects": objects_data,
            "component_breakdown": component_breakdown,
            "metric_explanations_cn": metric_explanations_cn(),
            "validation_framework_cn": build_framework_rows(),
            "notes_cn": {
                "v4_new": "v4增加了3D Print/CAD可用性相关检查：错误连续边、相交面、零长度边、非平面面、闭合规则边环估计、UV接缝/空间利用率人工评分。",
                "uv_overlap": "UV重叠为近似检测。若模型很大且超过三角面上限，会跳过；中建议结合UV编辑器截图复核。",
                "self_intersection": "相交面为BVH近似检测，已排除共享顶点的邻接三角面，但复杂模型仍建议人工复核。",
                "edge_loop": "闭合规则边环为近似估计，不等于严格的美术边环定义；请结合线框图和人工评分使用。",
                "architecture_accuracy": "门窗数量、开洞完整率、构件穿插/缝隙属于建筑语义指标，建议参考图纸或人工观察后填写。",
                "construction_time": "Blender无法从最终模型反推原始建模时间，相关时间需手动填写。",
                "elapsed_seconds": r6(elapsed),
            }
        }

        outdir = ensure_dir(props.output_dir)
        tag = clean_filename(props.project_tag)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"model_metrics_{tag}_{ts}"

        json_path = os.path.join(outdir, base + "_完整数据.json")
        latest_json_path = os.path.join(outdir, f"model_metrics_{tag}_latest_完整数据.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 中文汇总CSV
        summary_rows = []
        cn_map = {
            "project_tag": "项目标签",
            "blend_file": "Blender文件路径",
            "export_time": "导出时间",
            "blender_version": "Blender版本",
            "file_size_mb": "文件大小_MB",
            "object_count_scene_total": "场景对象总数",
            "mesh_object_count_scene_total": "场景Mesh对象总数",
            "object_count_analyzed": "本次分析Mesh对象数",
            "vertex_count_total": "顶点总数",
            "edge_count_total": "边总数",
            "face_count_total": "面总数",
            "tri_face_ratio_total": "三角面比例",
            "quad_face_ratio_total": "四边面比例",
            "nonmanifold_edge_count_total": "非流形边总数",
            "boundary_edge_count_total": "边界边/开放边总数",
            "bad_contiguous_edge_count_total": "错误连续边总数",
            "intersecting_triangle_pair_count_total": "相交三角面对总数",
            "zero_area_face_count_total": "零面积面总数",
            "zero_length_edge_count_total": "零长度边总数",
            "nonflat_face_count_total": "非平面面总数",
            "thin_triangle_ratio_total": "细长三角形比例",
            "vertex_valence_4_ratio_mean": "顶点4价比例均值",
            "estimated_closed_regular_edge_loop_count_total": "闭合规则边环估计数",
            "uv_island_count_total": "UV岛总数",
            "uv_overlap_pairs_total": "UV重叠对总数",
            "uv_degenerate_face_count_total": "UV退化面总数",
            "uv_outside_0_1_loop_count_total": "超出0-1空间UV Loop数",
            "uv_relative_texel_density_cv_mean": "UV Texel Density离散系数均值",
            "used_material_count_total": "实际使用材质总数",
            "missing_external_image_count": "缺失贴图数量",
            "named_mesh_object_ratio": "对象命名比例",
            "architectural_category_named_ratio": "建筑构件识别比例",
            "one_piece_mesh_warning": "一体化网格警告",
            "unapplied_scale_object_count": "未应用缩放对象数",
            "manual_total_minutes": "总制作时间_分钟",
            "manual_edit_task_average_minutes": "编辑任务平均时间_分钟",
        }
        for k, v in summary.items():
            summary_rows.append({"英文原字段": k, "中文指标": cn_map.get(k, k), "数值": v})
        summary_csv = os.path.join(outdir, base + "_中文汇总.csv")
        latest_summary_csv = os.path.join(outdir, f"model_metrics_{tag}_latest_中文汇总.csv")
        write_csv(summary_csv, summary_rows, ["英文原字段", "中文指标", "数值"])
        write_csv(latest_summary_csv, summary_rows, ["英文原字段", "中文指标", "数值"])

        # 对象明细CSV
        obj_headers = sorted(set(k for row in objects_data for k in row.keys()))
        obj_csv = os.path.join(outdir, base + "_对象明细.csv")
        write_csv(obj_csv, objects_data, obj_headers)

        # 评价框架CSV
        framework_csv = os.path.join(outdir, base + "_评价框架.csv")
        write_csv(framework_csv, build_framework_rows(), ["评价模块", "关键指标", "适用意义", "数据来源"])

        # 构件分类CSV
        comp_csv = os.path.join(outdir, base + "_构件分类.csv")
        write_csv(comp_csv, component_breakdown, ["category_cn", "object_count", "ratio"])

        # 指标说明CSV
        exp_csv = os.path.join(outdir, base + "_指标说明.csv")
        write_csv(exp_csv, metric_explanations_cn(), ["category", "metric", "explanation", "source"])

        self.report({'INFO'}, "模型评估数据已导出：" + outdir)
        return {'FINISHED'}

# -----------------------------
# UI面板
# -----------------------------

class ARCHMETRICS_V4_PT_MainPanel(Panel):
    bl_label = "建筑模型评估 v4"
    bl_idname = "ARCHMETRICS_V4_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "建筑模型评估"

    def draw(self, context):
        layout = self.layout
        p = context.scene.archmetrics_v4_props

        box = layout.box()
        box.label(text="基础设置")
        box.prop(p, "project_tag")
        box.prop(p, "output_dir")
        row = box.row()
        row.prop(p, "selected_only")
        row.prop(p, "use_evaluated_mesh")

        box = layout.box()
        box.label(text="高级检测开关")
        box.prop(p, "enable_uv_overlap")
        if p.enable_uv_overlap:
            box.prop(p, "max_uv_overlap_triangles")
            box.prop(p, "uv_overlap_grid_size")
        box.prop(p, "enable_self_intersection_check")
        if p.enable_self_intersection_check:
            box.prop(p, "max_intersection_triangles")

        box = layout.box()
        box.label(text="检测阈值")
        box.prop(p, "thin_triangle_aspect_threshold")
        box.prop(p, "close_vertex_threshold")
        box.prop(p, "zero_area_face_epsilon")
        box.prop(p, "zero_length_edge_epsilon")
        box.prop(p, "nonflat_face_tolerance")

        box = layout.box()
        box.label(text="构建时间 / min")
        box.prop(p, "initial_modeling_minutes")
        box.prop(p, "regeneration_or_revision_minutes")
        box.prop(p, "cleanup_minutes")
        box.prop(p, "topology_fix_minutes")
        box.prop(p, "uv_minutes")
        box.prop(p, "material_minutes")
        box.prop(p, "export_test_minutes")

        box = layout.box()
        box.label(text="建筑准确性（参考图/人工填写）")
        row = box.row(); row.prop(p, "reference_size_x"); row.prop(p, "reference_size_y"); row.prop(p, "reference_size_z")
        row = box.row(); row.prop(p, "reference_door_count"); row.prop(p, "detected_correct_door_count")
        row = box.row(); row.prop(p, "reference_window_count"); row.prop(p, "detected_correct_window_count")
        row = box.row(); row.prop(p, "reference_opening_count"); row.prop(p, "correct_opening_count")
        row = box.row(); row.prop(p, "component_intersection_count"); row.prop(p, "component_gap_count"); row.prop(p, "z_fighting_count")

        box = layout.box()
        box.label(text="可编辑性任务时间 / min")
        box.prop(p, "move_window_minutes")
        box.prop(p, "delete_door_fill_wall_minutes")
        box.prop(p, "change_wall_height_minutes")
        box.prop(p, "replace_roof_material_minutes")

        box = layout.box()
        box.label(text="拓扑/UV人工复核评分")
        box.prop(p, "manual_edge_loop_placement_score")
        box.prop(p, "manual_uv_overlap_percent")
        box.prop(p, "manual_uv_stretching_score")
        box.prop(p, "manual_texture_application_ready_score")
        box.prop(p, "manual_texture_space_utilization_score")
        box.prop(p, "manual_seam_placement_score")

        box = layout.box()
        box.label(text="应用准备/装配测试")
        row = box.row(); row.prop(p, "rigging_test_total"); row.prop(p, "rigging_success_count")
        box.prop(p, "deformation_error_count")
        box.prop(p, "deformation_error_note")

        layout.separator()
        layout.operator("archmetrics_v4.export_metrics_cn", icon="EXPORT")

classes = (
    ARCHMETRICS_V4_Properties,
    ARCHMETRICS_V4_OT_Export,
    ARCHMETRICS_V4_PT_MainPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.archmetrics_v4_props = PointerProperty(type=ARCHMETRICS_V4_Properties)


def unregister():
    if hasattr(bpy.types.Scene, "archmetrics_v4_props"):
        del bpy.types.Scene.archmetrics_v4_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
