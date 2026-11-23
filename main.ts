// Push-optimal Sokoban solver.
//
// The levels are built as independent rooms off one hallway, and a level's
// push optimum is the sum of its rooms' optima, so the solver splits the
// board at the hallway row and solves each room on its own with an A* over
// pushes: the heuristic is a minimum-cost box-to-goal matching over exact
// single-box push distances, with static dead squares and freeze deadlocks
// pruned. Rooms are finished one at a time; every room but the last must end
// with the player able to walk back out of its door. If a level does not
// split cleanly, the same push search runs on the whole board instead, and
// if the optimal pass ever exhausts its fixed work budget a weighted pass
// still returns a solving line when it finds one. All budgets count expanded
// states, never the clock, so two runs agree byte for byte.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];
const INF = 1e9;

// Minimum-cost perfect matching (Hungarian method, O(n^3)) on a square cost
// matrix held in a flat array; INF when no finite matching exists.
function hungarian(cost, n, u, v, p, way, minv, used) {
  u.fill(0); v.fill(0); p.fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    minv.fill(Infinity);
    used.fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  return -v[0];
}

// A push-level A* restricted to a region. playerAllowed marks the squares
// the player may stand on, boxAllowed the squares a box may be pushed onto;
// exitCell, when given, must be walk-reachable in the final state. Returns
// the move string, or null when the budget runs out or no solution exists.
function pushSearch(ctx, boxes0, player0, exitCell, weight, cap) {
  const { w, h, N, wall, isGoal, goals, playerAllowed, boxAllowed } = ctx;
  const nb = boxes0.length;
  const ng = goals.length;
  // Exact single-box push distances to each goal, pulled backwards: a box
  // moves from `from` to `c` only when the player can stand beyond it.
  const pdist = [];
  for (const g of goals) {
    const d = new Int32Array(N).fill(INF);
    d[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi];
      const cx = c % w, cy = (c / w) | 0;
      for (const [dx, dy] of DIRS) {
        const fx = cx - dx, fy = cy - dy, sx = cx - 2 * dx, sy = cy - 2 * dy;
        if (fx < 0 || fy < 0 || fx >= w || fy >= h || sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        const from = fy * w + fx, stand = sy * w + sx;
        if (!boxAllowed[from] || !playerAllowed[stand]) continue;
        if (d[from] === INF) { d[from] = d[c] + 1; queue.push(from); }
      }
    }
    pdist.push(d);
  }
  const dead = new Uint8Array(N);
  for (let c = 0; c < N; c++) {
    if (!boxAllowed[c] || isGoal[c]) continue;
    let alive = false;
    for (let gi = 0; gi < ng; gi++) if (pdist[gi][c] < INF) { alive = true; break; }
    if (!alive) dead[c] = 1;
  }
  const cost = new Float64Array(nb * ng);
  const hu = new Float64Array(nb + 1), hv = new Float64Array(nb + 1);
  const hp = new Int32Array(nb + 1), hway = new Int32Array(nb + 1);
  const hminv = new Float64Array(nb + 1), hused = new Array(nb + 1).fill(false);
  function heuristic(boxes) {
    for (let bi = 0; bi < nb; bi++) {
      const b = boxes[bi];
      let mn = INF;
      for (let gi = 0; gi < ng; gi++) {
        const c = pdist[gi][b];
        cost[bi * ng + gi] = c;
        if (c < mn) mn = c;
      }
      if (mn >= INF) return INF;
    }
    const m = hungarian(cost, nb, hu, hv, hp, hway, hminv, hused);
    return m >= INF ? INF : m;
  }
  function frozenDead(boxes) {
    for (let y = 0; y + 1 < h; y++) {
      for (let x = 0; x + 1 < w; x++) {
        const cells = [y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1];
        let solid = true, badBox = false;
        for (const c of cells) {
          if (wall[c]) continue;
          if (boxes.includes(c)) { if (!isGoal[c]) badBox = true; continue; }
          solid = false; break;
        }
        if (solid && badBox) return true;
      }
    }
    const boxSet = new Set(boxes);
    const frozen = new Set();
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of boxes) {
        if (frozen.has(b)) continue;
        const blockedT = (c) => c < 0 || c >= N || wall[c] || !boxAllowed[c] || frozen.has(c);
        const blockedS = (c) => c < 0 || c >= N || wall[c] || !playerAllowed[c] || frozen.has(c);
        const canMove = (dx, dy) => !blockedT(b + dy * w + dx) && !blockedS(b - (dy * w + dx));
        if (!canMove(-1, 0) && !canMove(1, 0) && !canMove(0, -1) && !canMove(0, 1)) {
          frozen.add(b); changed = true;
        } else if ((!canMove(-1, 0) && !canMove(1, 0)) || (!canMove(0, -1) && !canMove(0, 1))) {
          // One axis fully blocked: frozen only if the other axis is too.
          const hF = !canMove(-1, 0) && !canMove(1, 0);
          const vF = !canMove(0, -1) && !canMove(0, 1);
          if (hF && vF) { frozen.add(b); changed = true; }
        }
      }
    }
    for (const b of frozen) if (!isGoal[b]) return true;
    return false;
  }
  const prevP = new Int32Array(N);
  const prevN = new Int32Array(N);
  const queue = new Int32Array(N);
  function walkFrom(start, boxSet, prev) {
    prev.fill(-2);
    prev[start] = -1;
    let qh = 0, qt = 0;
    queue[qt++] = start;
    while (qh < qt) {
      const c = queue[qh++];
      const cx = c % w, cy = (c / w) | 0;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di];
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (prev[n] !== -2 || !playerAllowed[n] || boxSet.has(n)) continue;
        prev[n] = c;
        queue[qt++] = n;
      }
    }
    return prev;
  }
  function regionRep(prev) {
    for (let c = 0; c < N; c++) if (prev[c] !== -2) return c;
    return -1;
  }
  function walkPath(from, to, prev) {
    let s = '';
    let c = to;
    while (c !== from) {
      const p = prev[c];
      const d = c - p;
      s = (d === 1 ? 'r' : d === -1 ? 'l' : d === w ? 'd' : 'u') + s;
      c = p;
    }
    return s;
  }
  const open = [];
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && a[1] > b[1]);
  const hpush = (e) => {
    let i = open.length;
    open.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(open[i], open[p])) break;
      [open[i], open[p]] = [open[p], open[i]];
      i = p;
    }
  };
  const hpop = () => {
    const top = open[0];
    const last = open.pop();
    if (open.length > 0) {
      open[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= open.length) break;
        if (c + 1 < open.length && less(open[c + 1], open[c])) c++;
        if (!less(open[c], open[i])) break;
        [open[i], open[c]] = [open[c], open[i]];
        i = c;
      }
    }
    return top;
  };
  const h0 = heuristic(boxes0);
  if (h0 >= INF) return null;
  const sBoxes = [boxes0], sPlayer = [player0], sParent = [-1], sSeg = [''];
  const best = new Map();
  const keyOf = (boxes, rep) => rep + '|' + boxes.join(',');
  walkFrom(player0, new Set(boxes0), prevP);
  best.set(keyOf(boxes0, regionRep(prevP)), 0);
  hpush([weight * h0, 0, 0]);
  let expanded = 0;
  while (open.length > 0) {
    const [, g, s] = hpop();
    const boxes = sBoxes[s], player = sPlayer[s];
    const boxSet = new Set(boxes);
    walkFrom(player, boxSet, prevP);
    if (best.get(keyOf(boxes, regionRep(prevP))) !== g) continue;
    let solved = true;
    for (const b of boxes) if (!isGoal[b]) { solved = false; break; }
    if (solved && (exitCell === null || prevP[exitCell] !== -2)) {
      let moves = '';
      for (let t = s; t > 0; t = sParent[t]) moves = sSeg[t] + moves;
      return moves;
    }
    if (++expanded > cap) return null;
    for (let bi = 0; bi < nb; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < 4; di++) {
        const [dx, dy, letter] = DIRS[di];
        const stand = b - (dy * w + dx);
        const target = b + (dy * w + dx);
        if (stand < 0 || stand >= N || target < 0 || target >= N) continue;
        if (prevP[stand] === -2) continue;
        if (!boxAllowed[target] || boxSet.has(target)) continue;
        if (dead[target]) continue;
        const nboxes = boxes.slice();
        nboxes[bi] = target;
        nboxes.sort((a, b2) => a - b2);
        if (frozenDead(nboxes)) continue;
        const nh = heuristic(nboxes);
        if (nh >= INF) continue;
        const seg = walkPath(player, stand, prevP) + letter.toUpperCase();
        walkFrom(b, new Set(nboxes), prevN);
        const k = keyOf(nboxes, regionRep(prevN));
        const ng2 = g + 1;
        const seen = best.get(k);
        if (seen !== undefined && seen <= ng2) continue;
        best.set(k, ng2);
        const t = sBoxes.length;
        sBoxes.push(nboxes);
        sPlayer.push(b);
        sParent.push(s);
        sSeg.push(seg);
        hpush([ng2 + weight * nh, ng2, t]);
      }
    }
  }
  return null;
}

function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const N = w * h;
  const wall = new Uint8Array(N);
  const isGoal = new Uint8Array(N);
  const goals = [];
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ch = x < rows[y].length ? rows[y][x] : '#';
      const i = y * w + x;
      if (ch === '#') wall[i] = 1;
      if (ch === '.' || ch === '*' || ch === '+') { isGoal[i] = 1; goals.push(i); }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, N, wall, isGoal, goals, boxes, player };
}

// Walk the player over the whole board (no pushes) from `from` to `to`,
// avoiding walls and the given box set; returns the move string or null.
function boardWalk(L, from, to, boxSet) {
  const { w, h, N, wall } = L;
  const prev = new Int32Array(N).fill(-2);
  prev[from] = -1;
  const queue = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    if (c === to) break;
    const cx = c % w, cy = (c / w) | 0;
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (prev[n] !== -2 || wall[n] || boxSet.has(n)) continue;
      prev[n] = c;
      queue.push(n);
    }
  }
  if (prev[to] === -2) return null;
  let s = '';
  let c = to;
  while (c !== from) {
    const p = prev[c];
    const d = c - p;
    s = (d === 1 ? 'r' : d === -1 ? 'l' : d === w ? 'd' : 'u') + s;
    c = p;
  }
  return s;
}

// Replay moves on a mutable copy of the level; returns the new state.
function applyMoves(L, boxes, player, moves) {
  const { w, wall } = L;
  const set = new Set(boxes);
  let p = player;
  const D = { u: [0, -1], d: [0, 1], l: [-1, 0], r: [1, 0] };
  for (const ch of moves) {
    const [dx, dy] = D[ch.toLowerCase()];
    const q = p + dy * w + dx;
    if (set.has(q)) {
      set.delete(q);
      set.add(q + dy * w + dx);
    }
    p = q;
  }
  return { boxes: [...set].sort((a, b) => a - b), player: p };
}

function solveByRooms(L) {
  const { w, h, N, wall, isGoal } = L;
  const hallY = (L.player / w) | 0;
  // Rooms: the floor components left when the hallway row is walled off.
  const comp = new Int32Array(N).fill(-1);
  let ncomp = 0;
  for (let c = 0; c < N; c++) {
    if (wall[c] || ((c / w) | 0) === hallY || comp[c] !== -1) continue;
    const stack = [c];
    comp[c] = ncomp;
    while (stack.length > 0) {
      const cur = stack.pop();
      const cx = cur % w, cy = (cur / w) | 0;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || ny === hallY || comp[n] !== -1) continue;
        comp[n] = ncomp;
        stack.push(n);
      }
    }
    ncomp++;
  }
  const rooms = [];
  for (let ci = 0; ci < ncomp; ci++) {
    const cells = [], rboxes = [], rgoals = [], hallCells = [];
    for (let c = 0; c < N; c++) {
      if (comp[c] !== ci) continue;
      cells.push(c);
      if (L.boxes.includes(c)) rboxes.push(c);
      if (isGoal[c]) rgoals.push(c);
      const cx = c % w, cy = (c / w) | 0;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        if (ny === hallY && !wall[ny * w + nx]) hallCells.push(ny * w + nx);
      }
    }
    if (rboxes.length === 0 && rgoals.length === 0) continue;
    if (rboxes.length === 0 || rboxes.length !== rgoals.length || hallCells.length === 0) return null;
    rooms.push({ cells, boxes: rboxes, goals: rgoals, hall: hallCells[0] });
  }
  if (rooms.length === 0) return null;
  const totalBoxes = rooms.reduce((a, r) => a + r.boxes.length, 0);
  if (totalBoxes !== L.boxes.length) return null;
  rooms.sort((a, b) => (a.hall % w) - (b.hall % w));
  let boxes = L.boxes, player = L.player, out = '';
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const boxSet = new Set(boxes);
    const travel = boardWalk(L, player, room.hall, boxSet);
    if (travel === null) return null;
    out += travel;
    let st = applyMoves(L, boxes, player, travel);
    boxes = st.boxes; player = st.player;
    const playerAllowed = new Uint8Array(N);
    const boxAllowed = new Uint8Array(N);
    for (const c of room.cells) { playerAllowed[c] = 1; boxAllowed[c] = 1; }
    playerAllowed[room.hall] = 1;
    const ctx = { w, h, N, wall, isGoal, goals: room.goals, playerAllowed, boxAllowed };
    const last = ri === rooms.length - 1;
    const sol = pushSearch(ctx, room.boxes, player, last ? null : room.hall, 1, 300000);
    if (sol === null) return null;
    out += sol;
    st = applyMoves(L, boxes, player, sol);
    boxes = st.boxes; player = st.player;
  }
  return out;
}

function solveWhole(L) {
  const playerAllowed = new Uint8Array(L.N).fill(1);
  const boxAllowed = new Uint8Array(L.N).fill(1);
  for (let c = 0; c < L.N; c++) if (L.wall[c]) { playerAllowed[c] = 0; boxAllowed[c] = 0; }
  const ctx = { w: L.w, h: L.h, N: L.N, wall: L.wall, isGoal: L.isGoal, goals: L.goals, playerAllowed, boxAllowed };
  const optimal = pushSearch(ctx, L.boxes, L.player, null, 1, 300000);
  if (optimal !== null) return optimal;
  const greedy = pushSearch(ctx, L.boxes, L.player, null, 3, 300000);
  return greedy ?? '';
}

function solveLevel(text) {
  const L = parseLevel(text);
  if (L.boxes.length === 0) return '';
  try {
    const byRooms = solveByRooms(L);
    if (byRooms !== null) return byRooms;
  } catch {
    // fall through to the whole-board search
  }
  return solveWhole(L);
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
