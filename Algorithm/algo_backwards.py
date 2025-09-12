# grid_visit_obstacles_ui.py
# Python 3.9+
# Requires: numpy, matplotlib

import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import FancyArrow
from collections import deque
import json
import sys
import os

# =========================
# Dubins Path Planning Functions
# =========================
def _mod2pi(angle):
    """Normalize angle to [0, 2π)."""
    return angle % (2 * math.pi)

def _calc_trig_funcs(alpha, beta):
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_ab = math.cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab

def _LSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:  # invalid configuration
        return None, None, None, mode
    tmp = math.atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = math.sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode

def _RSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    tmp = math.atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = math.sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode

def _LSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = math.sqrt(p_squared)
    tmp = math.atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - math.atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode

def _RSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = math.sqrt(p_squared)
    tmp = math.atan2((cos_a + cos_b), (d - sin_a - sin_b)) - math.atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode

def _RLR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * math.pi - math.acos(tmp))
    d1 = _mod2pi(alpha - math.atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode

def _LRL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * math.pi - math.acos(tmp))
    d1 = _mod2pi(-alpha - math.atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode

def dubins_path(start_pose, end_pose, turning_radius):
    """
    Calculate Dubins path between two poses.
    Returns (d1, d2, d3, mode) or None.
    """
    x1, y1, theta1 = start_pose
    x2, y2, theta2 = end_pose
    
    dx = x2 - x1
    dy = y2 - y1
    d = math.sqrt(dx*dx + dy*dy) / turning_radius
    
    if d < 1e-6:  # Very close points
        alpha = 0
        beta = _mod2pi(theta2 - theta1)
    else:
        alpha = _mod2pi(math.atan2(dy, dx) - theta1)
        beta = _mod2pi(theta2 - math.atan2(dy, dx))
    
    paths = [
        _LSL(alpha, beta, d),
        _RSR(alpha, beta, d),
        _LSR(alpha, beta, d),
        _RSL(alpha, beta, d),
        _RLR(alpha, beta, d),
        _LRL(alpha, beta, d)
    ]
    
    best_path = None
    best_length = float('inf')
    for path in paths:
        if path[0] is not None:  # Valid path
            total_length = abs(path[0]) + abs(path[1]) + abs(path[2])
            if total_length < best_length:
                best_length = total_length
                best_path = path
    return best_path

def dubins_path_cost(start_pose, end_pose, turning_radius, speed):
    """
    Calculate the time cost of a Dubins path.
    """
    path = dubins_path(start_pose, end_pose, turning_radius)
    if path is None:
        return float('inf')
    d1, d2, d3, mode = path
    total_time = (abs(d1) + abs(d2) + abs(d3)) * turning_radius / speed
    return total_time

# =========================
# Grid / Robot parameters
# =========================
NCELLS = 20                 # 20x20 grid
CELL_CM = 10.0              # each cell = 10cm
WORLD_CM = NCELLS * CELL_CM

ROBOT_FOOTPRINT = 3         # robot is 3x3 cells
INFLATE_RADIUS = (ROBOT_FOOTPRINT - 1)//2  # = 1

OB_SIZE_CELLS = 1           # obstacle occupies 1x1 cell
SCAN_OFFSET_CELLS = 2       # 2 cells = 20cm away from obstacle side

# Start state (row, col, heading°); (0,0) is BOTTOM-LEFT; rows grow upward now.
# Headings are multiples of 90: 0=N, 90=E, 180=S, 270=W
START_RC = (1, 1, 0)

# Direction the ROBOT must face to scan a given obstacle side.
# (row, col) deltas expressed as the **final step into the scan cell**.
DIR_FOR_SIDE = {
    'N': (-1,  0),  # face South
    'S': (+1,  0),  # face North
    'E': ( 0, -1),  # face West
    'W': ( 0, +1),  # face East
}

# =========================
# Grid helpers
# =========================
def in_bounds(r, c):
    return 0 <= r < NCELLS and 0 <= c < NCELLS

def grid_with_obstacles(obstacles_rc):
    """Return a boolean grid (True=blocked) with raw obstacles."""
    grid = np.zeros((NCELLS, NCELLS), dtype=bool)
    for (r, c) in obstacles_rc:
        if in_bounds(r, c):
            grid[r, c] = True
    return grid

def inflate_blocked(grid, radius=INFLATE_RADIUS):
    """Morphological inflate: mark blocked any cell whose (2*radius+1)^2 neighborhood hits an obstacle."""
    if radius <= 0:
        return grid.copy()
    rows, cols = grid.shape
    rr = range(-radius, radius+1)
    neigh = [(dr, dc) for dr in rr for dc in rr]  # Chebyshev ball
    inflated = np.zeros_like(grid)
    obs = np.argwhere(grid)
    for (r, c) in obs:
        for dr, dc in neigh:
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < rows and 0 <= c2 < cols:
                inflated[r2, c2] = True
    return inflated

# =========================
# Scan target generation
# =========================
def scan_candidates_for_obstacle_cell(r, c, side, offset=SCAN_OFFSET_CELLS, lateral_span=1):
    side = side.upper()
    cands = []
    lat_offsets = [0] + [d for k in range(1, lateral_span+1) for d in (k, -k)]

    if side == 'N':
        base = (r + offset, c)   # was r - offset
        for d in lat_offsets:
            cands.append((base[0], base[1] + d))
    elif side == 'S':
        base = (r - offset, c)   # was r + offset
        for d in lat_offsets:
            cands.append((base[0], base[1] + d))
    elif side == 'E':
        base = (r, c + offset)
        for d in lat_offsets:
            cands.append((base[0] + d, base[1]))
    elif side == 'W':
        base = (r, c - offset)
        for d in lat_offsets:
            cands.append((base[0] + d, base[1]))
    else:
        raise ValueError("side must be one of 'N','S','E','W'")
    # Keep in-bounds & unique
    seen = set()
    out = []
    for (rr, cc) in cands:
        if in_bounds(rr, cc) and (rr, cc) not in seen:
            seen.add((rr, cc))
            out.append((rr, cc))
    # Fallback clamp if empty (edge cases near borders)
    if not out:
        rr, cc = base
        rr = min(max(rr, 0), NCELLS - 1)
        cc = min(max(cc, 0), NCELLS - 1)
        out = [(rr, cc)]
    return out

# =========================
# A* over (r,c,θ) with geometric/time costs
# =========================
HEADINGS = [0, 90, 180, 270]
DIRS = {
    0: (+1, 0),   # North (row up)
    90: (0, +1),  # East (col right)
    180: (-1, 0), # South (row down)
    270: (0, -1)  # West (col left)
}

# --- Motion / Time model ---
TURNING_RADIUS = 2.5   # cells (25 cm @ 10 cm/cell)
SPEED_CM_S = 20.0      # robot linear speed in cm/s

# Primitive **times** (seconds)
FORWARD_COST = CELL_CM / SPEED_CM_S
BACKWARD_COST = FORWARD_COST * 1.10
ARC_COST = (math.pi * TURNING_RADIUS * CELL_CM / 2.0) / SPEED_CM_S

RECOGNITION_TIME_S = 2.0
TIME_LIMIT_S = 120.0

SAMPLE_TRANSITIONS = True

def dir_to_theta(dr, dc):
    for theta, (rr, cc) in DIRS.items():
        if (rr, cc) == (dr, dc):
            return theta
    raise ValueError(f"No heading matches direction {(dr, dc)}")

def collision_free_cell(r, c, grid_blocked):
    return in_bounds(r, c) and (not grid_blocked[r, c])

def cells_between(a, b):
    """
    Conservative sampler of cells from a->b (ignores θ).
    """
    (ra, ca, _), (rb, cb, _) = a, b
    r, c = ra, ca
    dr = 1 if rb > ra else (-1 if rb < ra else 0)
    dc = 1 if cb > ca else (-1 if cb < ca else 0)
    cells = []
    while (r, c) != (rb, cb):
        if abs(rb - r) >= abs(cb - c):
            r += dr
        else:
            c += dc
        cells.append((r, c))
        if len(cells) > 4:
            break
    return cells

def transition_collision_free(a, b, grid_blocked):
    """Conservative collision check along the a->b transition."""
    if not collision_free_cell(b[0], b[1], grid_blocked):
        return False
    if not SAMPLE_TRANSITIONS:
        return True
    for (r, c) in cells_between(a, b):
        if not collision_free_cell(r, c, grid_blocked):
            return False
    return True

def motion_primitives(state):
    """
    Short nonholonomic moves with **time** costs (seconds).
    Includes forward/backward straights and forward/backward 90° arcs.
    """
    r, c, theta = state
    moves = []

    # forward / backward 1 cell
    dr, dc = DIRS[theta]
    fwd = (r + dr, c + dc, theta)
    bwd = (r - dr, c - dc, theta)
    moves.append((fwd, FORWARD_COST))
    moves.append((bwd, BACKWARD_COST))

    # FORWARD ARCS: left/right 90° arc (advance 1 cell forward and yaw ±90°)
    theta_l = (theta - 90) % 360
    theta_r = (theta + 90) % 360
    dr_l, dc_l = DIRS[theta_l]
    dr_r, dc_r = DIRS[theta_r]

    # Forward arc endpoints: one forward + one lateral cell
    fwd_left_end  = (r + dr + dr_l, c + dc + dc_l, theta_l)
    fwd_right_end = (r + dr + dr_r, c + dc + dc_r, theta_r)

    # BACKWARD ARCS: left/right 90° arc (reverse 1 cell and yaw ±90°)
    bwd_left_end  = (r - dr - dr_l, c - dc - dc_l, theta_l)
    bwd_right_end = (r - dr - dr_r, c - dc - dc_r, theta_r)

    # Calculate Dubins costs for all arcs
    start_x = c * CELL_CM
    start_y = r * CELL_CM
    start_theta_rad = math.radians(theta)
    
    # Forward left arc
    fwd_left_x = (c + dc + dc_l) * CELL_CM
    fwd_left_y = (r + dr + dr_l) * CELL_CM
    fwd_left_theta_rad = math.radians(theta_l)
    fwd_left_cost = dubins_path_cost(
        (start_x, start_y, start_theta_rad),
        (fwd_left_x, fwd_left_y, fwd_left_theta_rad),
        TURNING_RADIUS * CELL_CM,
        SPEED_CM_S
    )
    
    # Forward right arc
    fwd_right_x = (c + dc + dc_r) * CELL_CM
    fwd_right_y = (r + dr + dr_r) * CELL_CM
    fwd_right_theta_rad = math.radians(theta_r)
    fwd_right_cost = dubins_path_cost(
        (start_x, start_y, start_theta_rad),
        (fwd_right_x, fwd_right_y, fwd_right_theta_rad),
        TURNING_RADIUS * CELL_CM,
        SPEED_CM_S
    )
    
    # Backward left arc
    bwd_left_x = (c - dc - dc_l) * CELL_CM
    bwd_left_y = (r - dr - dr_l) * CELL_CM
    bwd_left_theta_rad = math.radians(theta_l)
    bwd_left_cost = dubins_path_cost(
        (start_x, start_y, start_theta_rad),
        (bwd_left_x, bwd_left_y, bwd_left_theta_rad),
        TURNING_RADIUS * CELL_CM,
        SPEED_CM_S
    )
    
    # Backward right arc
    bwd_right_x = (c - dc - dc_r) * CELL_CM
    bwd_right_y = (r - dr - dr_r) * CELL_CM
    bwd_right_theta_rad = math.radians(theta_r)
    bwd_right_cost = dubins_path_cost(
        (start_x, start_y, start_theta_rad),
        (bwd_right_x, bwd_right_y, bwd_right_theta_rad),
        TURNING_RADIUS * CELL_CM,
        SPEED_CM_S
    )
    
    # Add all arc moves with their costs
    moves.append((fwd_left_end, fwd_left_cost if fwd_left_cost != float('inf') else ARC_COST))
    moves.append((fwd_right_end, fwd_right_cost if fwd_right_cost != float('inf') else ARC_COST))
    moves.append((bwd_left_end, bwd_left_cost if bwd_left_cost != float('inf') else ARC_COST))
    moves.append((bwd_right_end, bwd_right_cost if bwd_right_cost != float('inf') else ARC_COST))

    return moves

def _angle_quarters(a_deg):
    """Minimum number of 90° quarter-turns to rotate by a_deg (0..180)."""
    a = a_deg % 360
    a = min(a, 360 - a)
    return int(math.ceil(a / 90.0 - 1e-12))

def astar(grid_blocked, start, goal_rc, goal_theta=None):
    """
    Optimal A* over (r,c,theta) with primitive **time** costs.
    Heuristic:
      straight-line time lower bound + (#quarter-turns still needed)*ARC_COST.
    """
    sr, sc, stheta = start
    gr, gc = goal_rc
    if not in_bounds(sr, sc) or not in_bounds(gr, gc):
        return []
    if grid_blocked[sr, sc] or grid_blocked[gr, gc]:
        return []

    def h(state):
        r, c, theta = state
        dist_cells = math.hypot(gr - r, gc - c)
        dist_time = (dist_cells * CELL_CM) / SPEED_CM_S
        if goal_theta is None:
            turn_lb = 0.0
        else:
            diff = abs(goal_theta - theta) % 360
            diff = min(diff, 360 - diff)
            quarters = _angle_quarters(diff)
            turn_lb = quarters * ARC_COST
        return dist_time + turn_lb

    def f_key(g, state):
        return g + h(state) + 1e-6 * h(state)

    openq = []
    g_cost = {start: 0.0}
    parent = {start: None}
    push_id = 0
    heapq.heappush(openq, (f_key(0.0, start), push_id, start))

    closed_best_g = {}

    while openq:
        _, _, cur = heapq.heappop(openq)
        gcurr = g_cost[cur]

        if cur in closed_best_g and gcurr >= closed_best_g[cur] - 1e-12:
            continue
        closed_best_g[cur] = gcurr

        if (cur[0], cur[1]) == (gr, gc) and (goal_theta is None or cur[2] == goal_theta):
            path = []
            k = cur
            while k is not None:
                path.append(k)
                k = parent[k]
            return path[::-1]

        for nxt, step_cost in motion_primitives(cur):
            if not transition_collision_free(cur, nxt, grid_blocked):
                continue
            ng = gcurr + step_cost
            if ng + 1e-12 < g_cost.get(nxt, float('inf')):
                g_cost[nxt] = ng
                parent[nxt] = cur
                push_id += 1
                heapq.heappush(openq, (f_key(ng, nxt), push_id, nxt))

    return []

# =========================
# Time-aware helpers
# =========================
def _step_cost(a, b):
    """
    Determine primitive **time** between two consecutive (r,c,theta) states.
    Handles forward/backward straights and forward/backward arcs.
    """
    ra, ca, ta = a
    rb, cb, tb = b
    
    # Same heading - straight movement
    if ta == tb:
        dr, dc = rb - ra, cb - ca
        fdr, fdc = DIRS[ta]
        if (dr, dc) == (fdr, fdc):
            return FORWARD_COST
        if (dr, dc) == (-fdr, -fdc):
            return BACKWARD_COST
    
    # Different heading - check for arc movement
    if (tb - ta) % 360 in (90, 270):  # 90° turn
        # Get direction vectors
        fdr_a, fdc_a = DIRS[ta]
        fdr_b, fdc_b = DIRS[tb]
        
        # Forward arc: advance in original direction + lateral in new direction
        if (rb - ra, cb - ca) == (fdr_a + fdr_b, fdc_a + fdc_b):
            return ARC_COST
        
        # Backward arc: reverse in original direction + lateral in new direction  
        if (rb - ra, cb - ca) == (-fdr_a - fdr_b, -fdc_a - fdc_b):
            return ARC_COST
    
    # Fallback to forward cost
    return FORWARD_COST

def path_cost(states):
    """Sum **times** over a sequence of states (r,c,theta)."""
    if not states or len(states) == 1:
        return 0.0
    total = 0.0
    for i in range(len(states)-1):
        total += _step_cost(states[i], states[i+1])
    return total

def astar_with_cost(grid_blocked, start, goal_rc, goal_theta=None):
    """
    Run A* and return (states_list, time_seconds). Empty states => ([], inf)
    """
    seg = astar(grid_blocked, start, goal_rc, goal_theta)
    if not seg:
        return [], float('inf')
    return seg, path_cost(seg)

def _turn_dir(ta, tb):
    """Return 'RIGHT' if tb = ta+90, 'LEFT' if tb = ta-90, else None."""
    d = (tb - ta) % 360
    if d == 90:
        return 'RIGHT'
    if d == 270:
        return 'LEFT'
    return None

def _primitive_from_edge(a, b):
    """
    Classify a single edge (a->b) from full_path into a primitive.
    Now handles backward arcs as well.
    """
    ra, ca, ta = a
    rb, cb, tb = b

    # Forward/back straights
    if ta == tb:
        fdr, fdc = DIRS[ta]
        if (rb - ra, cb - ca) == (fdr, fdc):
            return {'type':'FWD', 'cells':1, 'dt': FORWARD_COST,
                    'from': (ra, ca, ta), 'to': (rb, cb, tb)}
        if (rb - ra, cb - ca) == (-fdr, -fdc):
            return {'type':'BWD', 'cells':1, 'dt': BACKWARD_COST,
                    'from': (ra, ca, ta), 'to': (rb, cb, tb)}

    # Check for 90° arc movements
    tdir = _turn_dir(ta, tb)
    if tdir is not None:
        fdr_a, fdc_a = DIRS[ta]
        fdr_b, fdc_b = DIRS[tb]
        
        # Forward arc
        if (rb - ra, cb - ca) == (fdr_a + fdr_b, fdc_a + fdc_b):
            return {'type':'ARC_FWD', 'direction': tdir,
                    'advance_cells': 1, 'delta_heading_deg': 90,
                    'dt': ARC_COST,
                    'from': (ra, ca, ta), 'to': (rb, cb, tb)}
        
        # Backward arc
        if (rb - ra, cb - ca) == (-fdr_a - fdr_b, -fdc_a - fdc_b):
            return {'type':'ARC_BWD', 'direction': tdir,
                    'advance_cells': -1, 'delta_heading_deg': 90,
                    'dt': ARC_COST,
                    'from': (ra, ca, ta), 'to': (rb, cb, tb)}

    # Fallback
    return {'type':'FWD', 'cells':1, 'dt': FORWARD_COST,
            'from': (ra, ca, ta), 'to': (rb, cb, tb)}

def _merge_linear_steps(steps):
    """
    Merge consecutive FWD/BWD into single steps.
    """
    out = []
    for s in steps:
        if out and s['type'] in ('FWD','BWD') and out[-1]['type'] == s['type']:
            out[-1]['cells'] += s['cells']
            out[-1]['dt'] += s['dt']
            out[-1]['to'] = s['to']
        else:
            out.append(dict(s))
    return out

def movements_from_path(full_path, breaks, scans_rc, time_limit=TIME_LIMIT_S):
    """
    Build a NAVIGATION payload with commands + path.
    Now supports backward arc movements (LB090, RB090).
    """
    if not full_path:
        return {'meta':{}, 'steps':[], 'totals':{}}, ""

    # 1) raw primitives
    raw = []
    for i in range(len(full_path)-1):
        raw.append(_primitive_from_edge(full_path[i], full_path[i+1]))

    # 2) insert RECOGNIZE events after indices in 'breaks'
    break_set = set(breaks)
    with_recog = []
    for i, step in enumerate(raw):
        with_recog.append(step)
        arrive_idx = i+1
        if arrive_idx in break_set:
            with_recog.append({
                'type':'RECOGNIZE',
                'dt': RECOGNITION_TIME_S,
                'at': full_path[arrive_idx],
            })

    # 3) merge linear (only merge consecutive FWD/BWD of same type)
    merged = _merge_linear_steps(with_recog)

    # 4) accumulate time, trim at limit
    t = 0.0
    steps_out = []
    recognized_count = 0
    for s in merged:
        dt = s['dt']
        if t + dt > time_limit + 1e-9:
            if s['type'] in ('FWD','BWD') and dt > 0:
                frac = max(0.0, (time_limit - t) / dt)
                if frac > 1e-6:
                    partial_cells = s['cells'] * frac
                    dist = int(round(partial_cells * CELL_CM))
                    move_code = f"SF{dist:03d}" if s['type'] == 'FWD' else f"SB{dist:03d}"
                    steps_out.append({
                        'type': s['type'],
                        'cells': round(partial_cells, 3),
                        'dt': (time_limit - t),
                        'from': s.get('from'),
                        'to': s.get('to'),
                        'move_code': move_code,
                    })
                    t = time_limit
            break

        new_s = dict(s)
        t += dt
        if s['type'] == 'RECOGNIZE':
            recognized_count += 1
        elif s['type'] == 'FWD':
            dist = int(round(s['cells'] * CELL_CM))
            new_s['move_code'] = f"SF{dist:03d}"
        elif s['type'] == 'BWD':
            dist = int(round(s['cells'] * CELL_CM))
            new_s['move_code'] = f"SB{dist:03d}"
        elif s['type'] == 'ARC_FWD':
            new_s['move_code'] = "LF090" if s['direction'] == 'LEFT' else "RF090"
        elif s['type'] == 'ARC_BWD':
            new_s['move_code'] = "LB090" if s['direction'] == 'LEFT' else "RB090"
        steps_out.append(new_s)

        if abs(t - time_limit) <= 1e-9:
            break

    tokens = []
    for s in steps_out:
        if s['type'] == 'FWD':
            dist = int(round(s['cells'] * CELL_CM))
            tokens.append(f"SF{dist:03d}")
        elif s['type'] == 'BWD':
            dist = int(round(s['cells'] * CELL_CM))
            tokens.append(f"SB{dist:03d}")
        elif s['type'] == 'ARC_FWD':
            tokens.append("LF090" if s['direction'] == 'LEFT' else "RF090")
        elif s['type'] == 'ARC_BWD':
            tokens.append("LB090" if s['direction'] == 'LEFT' else "RB090")
        elif s['type'] == 'RECOGNIZE':
            tokens.append("IMAGE_REC")

    path_coords = [[c, r] for (r, c, theta) in full_path]

    trace = {
        "type": "NAVIGATION",
        "data": {
            "commands": tokens,
            "path": path_coords
        }
    }
    return trace, tokens

def save_movement_trace(trace, path="movement_trace.json"):
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)
    return path

# =========================
# Multi-target routing (Hamiltonian shortest-time via Held–Karp)
# =========================
def _scan_goal_for_item(item):
    """From {"rc":(r,c), "side": 'N'|'S'|'E'|'W'} compute (goal_rc, goal_theta, approach_rc)."""
    (r, c) = item["rc"]
    side = item["side"].upper()
    cand_list = scan_candidates_for_obstacle_cell(r, c, side, offset=SCAN_OFFSET_CELLS, lateral_span=0)
    goal_rc = cand_list[0]  # guaranteed non-empty by helper
    dr_req, dc_req = DIR_FOR_SIDE[side]
    goal_theta = dir_to_theta(dr_req, dc_req)
    approach_rc = (goal_rc[0] - dr_req, goal_rc[1] - dc_req)
    return goal_rc, goal_theta, approach_rc

def plan_route_tsp(grid_blocked, start_state, obstacles_with_sides):
    """
    Build a shortest-time Hamiltonian path over image scan goals.
    """
    n = len(obstacles_with_sides)
    if n == 0:
        return [start_state], [], [], []

    # Build nodes
    nodes = []
    for item in obstacles_with_sides:
        goal_rc, goal_theta, approach_rc = _scan_goal_for_item(item)
        nodes.append({"goal": goal_rc, "theta": goal_theta, "approach": approach_rc})

    # Feasibility: discard nodes whose scan cell or approach is blocked/out-of-bounds
    feasible = []
    idx_map = []
    for i, nd in enumerate(nodes):
        gr, gc = nd["goal"]
        ar, ac = nd["approach"]
        if not (in_bounds(gr, gc) and in_bounds(ar, ac)):
            continue
        if grid_blocked[gr, gc] or grid_blocked[ar, ac]:
            continue
        feasible.append(nd)
        idx_map.append(i)

    if not feasible:
        return [start_state], [], [None]*n, []

    m = len(feasible)

    # From start -> each j
    INF = float('inf')
    cost_start = [INF]*m
    seg_start = [None]*m
    for j, nd in enumerate(feasible):
        seg, t = astar_with_cost(grid_blocked, start_state, nd["approach"], nd["theta"])
        if seg:
            cost_start[j] = t + FORWARD_COST
            seg_start[j] = seg

    # i -> j costs
    cost = [[INF]*m for _ in range(m)]
    seg_ij = [[None]*m for _ in range(m)]
    for i, ndi in enumerate(feasible):
        si = (ndi["goal"][0], ndi["goal"][1], ndi["theta"])
        for j, ndj in enumerate(feasible):
            if i == j:
                continue
            seg, t = astar_with_cost(grid_blocked, si, ndj["approach"], ndj["theta"])
            if seg:
                cost[i][j] = t + FORWARD_COST
                seg_ij[i][j] = seg

    # Held–Karp
    dp = { (1<<j, j): (cost_start[j], -1) for j in range(m) }
    for mask in range(1, 1<<m):
        for j in range(m):
            if not (mask & (1<<j)): continue
            key = (mask, j)
            if key not in dp or dp[key][0] == float('inf'): continue
            curr_cost, _ = dp[key]
            for k in range(m):
                if mask & (1<<k): continue
                step = cost[j][k]
                new_cost = curr_cost + step
                nk = (mask | (1<<k), k)
                if nk not in dp or new_cost < dp[nk][0]:
                    dp[nk] = (new_cost, j)

    full_mask = (1<<m) - 1
    best_end = None
    best_cost = float('inf')
    for j in range(m):
        key = (full_mask, j)
        if key in dp and dp[key][0] < best_cost:
            best_cost = dp[key][0]
            best_end = j

    if best_end is None or best_cost == float('inf'):
        order_feas = sorted(range(m), key=lambda j: cost_start[j])
    else:
        order_feas = []
        mask = full_mask
        j = best_end
        while j != -1:
            order_feas.append(j)
            _, pj = dp[(mask, j)]
            mask &= ~(1<<j)
            j = pj
        order_feas.reverse()

    # Build full path
    full_path = [start_state]
    breaks = []
    chosen_scans = [None]*len(obstacles_with_sides)
    visit_order = []

    cur_state = start_state
    for jf in order_feas:
        nd = feasible[jf]
        if cur_state == start_state and seg_start[jf] is not None:
            seg = seg_start[jf]
        else:
            seg, _ = astar_with_cost(grid_blocked, cur_state, nd["approach"], nd["theta"])
            if not seg:
                continue
        final_state = (nd["goal"][0], nd["goal"][1], nd["theta"])
        if not transition_collision_free(seg[-1], final_state, grid_blocked):
            continue

        full_path += seg[1:] + [final_state]
        breaks.append(len(full_path)-1)

        orig_idx = idx_map[jf]
        chosen_scans[orig_idx] = nd["goal"]
        visit_order.append(orig_idx)

        cur_state = final_state

    return full_path, breaks, chosen_scans, visit_order

# =========================
# Drawing helpers (0,0 at bottom-left)
# =========================
def draw_grid(ax):
    for r in range(NCELLS+1):
        ax.plot([0, NCELLS], [r, r], '-', lw=0.5, color='0.8', zorder=0)
    for c in range(NCELLS+1):
        ax.plot([c, c], [0, NCELLS], '-', lw=0.5, color='0.8', zorder=0)
    ax.plot([0, NCELLS, NCELLS, 0, 0], [0, 0, NCELLS, NCELLS, 0], 'k-', lw=1.0, zorder=1)

def cell_center_xy(rc):
    r, c = rc
    x = c + 0.5
    y = r + 0.5   # bottom-left origin
    return x, y

def draw_cell_square(ax, rc, color='tab:red', lw=2, alpha=1.0):
    r, c = rc
    x0, y0 = c, r  # bottom-left of the cell
    xs = [x0, x0+1, x0+1, x0, x0]
    ys = [y0, y0, y0+1, y0+1, y0]
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)

def draw_robot_footprint(ax, rc, color='k', lw=2):
    """Draw 3x3 robot footprint centered at rc."""
    r, c = rc
    r0, c0 = r - INFLATE_RADIUS, c - INFLATE_RADIUS
    x0, y0 = c0, r0
    xs = [x0, x0+ROBOT_FOOTPRINT, x0+ROBOT_FOOTPRINT, x0, x0]
    ys = [y0, y0, y0+ROBOT_FOOTPRINT, y0+ROBOT_FOOTPRINT, y0]
    ax.plot(xs, ys, color=color, lw=lw)

def draw_start_zone(ax):
    """
    Draw a shaded 3x3 START ZONE centered on START_RC (row,col).
    """
    sr, sc, _ = START_RC
    r0 = max(0, sr - INFLATE_RADIUS)
    c0 = max(0, sc - INFLATE_RADIUS)
    r1 = min(NCELLS-1, sr + INFLATE_RADIUS)
    c1 = min(NCELLS-1, sc + INFLATE_RADIUS)

    x0, y0 = c0, r0
    w = (c1 - c0 + 1)
    h = (r1 - r0 + 1)

    ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor='tab:blue', alpha=0.12, edgecolor='tab:blue', lw=1.5))
    xlab, ylab = cell_center_xy((sr, sc))
    ax.text(xlab, ylab, "START ZONE", color='tab:blue', fontsize=9, ha='center', va='center')

# =========================
# Animation (compatible with bottom-left origin)
# =========================
def animate_path(grid_blocked, obstacles_rc, scans_rc, visit_order, full_path, breaks):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(0, NCELLS); ax.set_ylim(0, NCELLS)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Shortest-time Hamiltonian path — camera-facing scans; orange = heading")

    draw_grid(ax)
    draw_start_zone(ax)

    # Obstacles
    for rc in obstacles_rc:
        draw_cell_square(ax, rc, color='tab:red', lw=2)

    # Scan targets + labels
    for i, sp in enumerate(scans_rc):
        if sp is None:
            continue
        x, y = cell_center_xy(sp)
        ax.plot(x, y, 'o', color='tab:green', markersize=6)
        ax.text(x+0.05, y+0.1, f"S{i+1}", color='tab:green', fontsize=9)

    # Visit order markers
    for rank, idx in enumerate(visit_order, 1):
        sp = scans_rc[idx]
        if sp is None:
            continue
        x, y = cell_center_xy(sp)
        ax.text(x-0.35, y-0.35, f"#{rank}", color='tab:blue', fontsize=9)
        ax.plot(x, y, 'x', color='tab:blue')

    # Path line (cell centers)
    path_xy = np.array([cell_center_xy((r, c)) for (r, c, theta) in full_path])
    (path_line,) = ax.plot([], [], '-', lw=2, alpha=0.9, color='tab:blue')

    # Robot marker (3x3) and heading arrow
    robot_outline, = ax.plot([], [], 'k-', lw=2)
    r0, c0, theta0 = full_path[0]
    x0, y0 = cell_center_xy((r0, c0))
    rad0 = math.radians(theta0)
    dx0 = 0.8 * math.sin(rad0)
    dy0 = 0.8 * math.cos(rad0)
    camera_arrow = FancyArrow(
        x0, y0, dx0, dy0,
        width=0.1, length_includes_head=True,
        head_width=0.4, head_length=0.4,
        color="orange"
    )
    ax.add_patch(camera_arrow)

    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left',
                  fontsize=10, color='black',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='0.7'))

    N = len(full_path)
    step_times = [0.0]*(N-1)
    for i in range(N-1):
        step_times[i] = _step_cost(full_path[i], full_path[i+1])

    break_set = set(breaks)
    cum_time = [0.0]*N
    for i in range(1, N):
        cum_time[i] = cum_time[i-1] + step_times[i-1]
        if i in break_set:
            cum_time[i] += RECOGNITION_TIME_S

    last_frame = 0
    for i in range(N):
        if cum_time[i] <= TIME_LIMIT_S + 1e-9:
            last_frame = i
        else:
            break

    breaks_sorted = sorted(breaks)
    def recognized_count_at(frame_idx):
        cnt = 0
        for b in breaks_sorted:
            if b <= frame_idx and cum_time[b] <= TIME_LIMIT_S + 1e-9:
                cnt += 1
        return cnt

    def robot_poly(rc):
        r, c = rc
        r0, c0 = r - INFLATE_RADIUS, c - INFLATE_RADIUS
        x0, y0 = c0, r0
        xs = [x0, x0+ROBOT_FOOTPRINT, x0+ROBOT_FOOTPRINT, x0, x0]
        ys = [y0, y0, y0+ROBOT_FOOTPRINT, y0+ROBOT_FOOTPRINT, y0]
        return xs, ys

    def init():
        path_line.set_data([], [])
        robot_outline.set_data([], [])
        hud.set_text("")
        return path_line, robot_outline, camera_arrow, hud

    def update(frame):
        nonlocal camera_arrow
        path_line.set_data(path_xy[:frame+1,0], path_xy[:frame+1,1])
        xs, ys = robot_poly(full_path[frame][:2])
        robot_outline.set_data(xs, ys)

        camera_arrow.remove()
        r, c, theta = full_path[frame]
        x, y = cell_center_xy((r, c))
        rad = math.radians(theta)
        dx = 0.8 * math.sin(rad)
        dy = 0.8 * math.cos(rad)
        camera_arrow = FancyArrow(
            x, y, dx, dy,
            width=0.1, length_includes_head=True,
            head_width=0.4, head_length=0.4,
            color="orange"
        )
        ax.add_patch(camera_arrow)

        t = cum_time[frame]
        recog = recognized_count_at(frame)
        hud.set_text(f"t = {t:0.1f} s   |   recognized: {recog}/{len([s for s in scans_rc if s is not None])}   |   limit = {TIME_LIMIT_S:.0f} s")

        return path_line, robot_outline, camera_arrow, hud

    total_frames = last_frame + 1
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        init_func=init, interval=120, blit=True, repeat=False
    )

    plt.show()

# =========================
# Click-to-place UI (bottom-left origin)
# =========================
class InteractivePlacer:
    def __init__(self):
        self.items = []   # list of dicts {"rc":(r,c), "side": 'N'|'S'|'E'|'W'}
        self.current_side = 'N'
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        self.ax.set_xlim(0, NCELLS); self.ax.set_ylim(0, NCELLS)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Click to place 1×1 obstacles (choose side). Press PLAN to compute path.")
        draw_grid(self.ax)
        draw_start_zone(self.ax)

        # UI panel
        plt.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.08)
        ax_radio = plt.axes([0.87, 0.55, 0.10, 0.25])
        self.radio = RadioButtons(ax_radio, ('N','S','E','W'))
        self.radio.on_clicked(self._on_side)

        ax_plan = plt.axes([0.87, 0.45, 0.10, 0.07])
        self.btn_plan = Button(ax_plan, 'PLAN')
        self.btn_plan.on_clicked(self._on_plan)

        ax_undo = plt.axes([0.87, 0.35, 0.10, 0.07])
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_undo.on_clicked(self._on_undo)

        ax_clear = plt.axes([0.87, 0.25, 0.10, 0.07])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self._on_clear)

        ax_info = plt.axes([0.86, 0.05, 0.12, 0.17]); ax_info.axis('off')
        self.info_text = ax_info.text(0, 1, self._status_text(), va='top')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.done = False
        self.result = None

    def _status_text(self):
        return (f"Side: {self.current_side}\n"
                f"Placed: {len(self.items)}\n"
                f"Robot: 3×3\n"
                f"Grid: 20×20")

    def _on_side(self, label):
        self.current_side = str(label).upper()
        self.info_text.set_text(self._status_text()); self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or self.done:
            return
        col = int(event.xdata)
        row = int(event.ydata)   # bottom-left origin
        r = row
        c = col
        if not in_bounds(r, c):
            return
        item = {"rc": (r, c), "side": self.current_side}
        self.items.append(item)
        draw_cell_square(self.ax, item["rc"], color='tab:red', lw=2)
        rr, cc = item["rc"]
        cand = scan_candidates_for_obstacle_cell(rr, cc, item["side"],
                                                 offset=SCAN_OFFSET_CELLS, lateral_span=0)[0]
        x, y = cell_center_xy(cand)
        self.ax.plot(x, y, 'o', color='tab:green', markersize=4, alpha=0.8)
        self.ax.text(*cell_center_xy(item["rc"]), item["side"],
                     color='tab:blue', fontsize=8, ha='center', va='center')
        self.info_text.set_text(self._status_text()); self.fig.canvas.draw_idle()

    def _on_undo(self, _):
        if not self.items or self.done:
            return
        self.items.pop()
        self.ax.clear()
        self.ax.set_xlim(0, NCELLS); self.ax.set_ylim(0, NCELLS)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Click to place 1×1 obstacles (choose side). Press PLAN to compute path.")
        draw_grid(self.ax)
        draw_start_zone(self.ax)
        for item in self.items:
            draw_cell_square(self.ax, item["rc"], color='tab:red', lw=2)
            rr, cc = item["rc"]
            cand = scan_candidates_for_obstacle_cell(rr, cc, item["side"],
                                                     offset=SCAN_OFFSET_CELLS, lateral_span=0)[0]
            x, y = cell_center_xy(cand)
            self.ax.plot(x, y, 'o', color='tab:green', markersize=4, alpha=0.8)
            self.ax.text(*cell_center_xy(item["rc"]), item["side"],
                         color='tab:blue', fontsize=8, ha='center', va='center')
        self.info_text.set_text(self._status_text()); self.fig.canvas.draw_idle()

    def _on_clear(self, _):
        if self.done:
            return
        self.items.clear()
        self.ax.clear()
        self.ax.set_xlim(0, NCELLS); self.ax.set_ylim(0, NCELLS)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Click to place 1×1 obstacles (choose side). Press PLAN to compute path.")
        draw_grid(self.ax)
        draw_start_zone(self.ax)
        self.info_text.set_text(self._status_text()); self.fig.canvas.draw_idle()

    def _on_plan(self, _):
        if not self.items:
            self.ax.set_title("Place at least 1 obstacle, then press PLAN.")
            self.fig.canvas.draw_idle()
            return
        self.done = True
        self.result = list(self.items)
        plt.close(self.fig)

    def run(self):
        plt.show()
        return self.result

# =========================
# Utilities
# =========================
def nearest_free_center(blocked, start_rc):
    """
    If start_rc is blocked, BFS to find the nearest free cell (by 4-connected distance).
    Returns a (r,c) where the 3×3 robot footprint is clear in the inflated grid.
    """
    if not blocked[start_rc]:
        return start_rc
    q = deque([start_rc])
    seen = {start_rc}
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            if not blocked[nr, nc]:
                return (nr, nc)
            q.append((nr, nc))
    return start_rc

# =========================
# Main (task1 accepts dict payload)
# =========================
def _items_from_payload(payload_dict):
    """
    Extract obstacles [{"rc":(r,c), "side":...}, ...] from a START_TASK-like dict.
    """
    items = []
    obstacles = payload_dict.get("data", {}).get("obstacles", [])
    for o in obstacles:
        r = int(o.get("y"))  # y = row
        c = int(o.get("x"))  # x = col
        side = str(o.get("side", "N")).upper()
        items.append({"rc": (r, c), "side": side})
    return items

def _start_override_from_payload(payload_dict):
    """
    Optionally extract robot start (r,c,theta) from payload_dict["data"]["robot"].
    dir: 'N','E','S','W' -> 0,90,180,270
    """
    robot = payload_dict.get("data", {}).get("robot", {})
    if not robot:
        return None
    dir_map = {"N": 0, "E": 90, "S": 180, "W": 270}
    try:
        rx = int(robot.get("x"))
        ry = int(robot.get("y"))
        rdir = str(robot.get("dir", "N")).upper()
        return (ry, rx, dir_map.get(rdir, START_RC[2]))
    except Exception:
        return None

def task1(json_payload=None):
    """
    Accepts:
      - dict  (START_TASK-style payload)
      - None  (opens interactive placer)
    Saves movement_trace.json next to this file.
    """
    # 1) Obstacles + sides
    if json_payload is None:
        placer = InteractivePlacer()
        items = placer.run()
        start_override = None
    elif isinstance(json_payload, dict):
        items = _items_from_payload(json_payload)
        start_override = _start_override_from_payload(json_payload)
        print(f"Loaded {len(items)} obstacles from dict payload")
    else:
        raise TypeError("task1 json_payload must be dict or None")

    if not items:
        print("No obstacles placed. Exiting.")
        return

    # 2) Build blocked grid
    obstacles_rc = [it["rc"] for it in items]
    sides = [it["side"] for it in items]
    raw_grid = grid_with_obstacles(obstacles_rc)
    blocked = inflate_blocked(raw_grid, radius=INFLATE_RADIUS)

    global START_RC
    # 3) Start state (override allowed) with auto-relocate if blocked
    r, c, theta = START_RC
    if start_override is not None:
        r, c, theta = start_override
        
        START_RC = (r, c, theta)

    if blocked[r, c]:
        new_start = nearest_free_center(blocked, (r, c))
        print(f"Start {(r, c)} is blocked after inflation; using nearest free start {new_start}.")
        r, c = new_start

    start_state = (r, c, theta)
    obstacles_with_sides = [{"rc": rc, "side": s} for rc, s in zip(obstacles_rc, sides)]
    full_path, breaks, scans, order = plan_route_tsp(blocked, start_state, obstacles_with_sides)

    # 4) Build + save movement JSON (save into Algorithm folder)
    trace, token_str = movements_from_path(full_path, breaks, scans, time_limit=TIME_LIMIT_S)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    outfile = save_movement_trace(trace, os.path.join(out_dir, "movement_trace.json"))
    print("\n=== MOVEMENT TOKENS ===")
    print(token_str)
    print(f"\nSaved JSON trace to: {outfile}")

    # 5) Animate if you want a UI (commented for headless use)
    animate_path(blocked, obstacles_rc, scans, order, full_path, breaks)

if __name__ == "__main__":
    # For manual testing you can still pass a JSON file path and we’ll load & run it.
    # (Kept for convenience in development.)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1], "r") as f:
            payload = json.load(f)
        task1(payload)
    else:
        task1(None)
