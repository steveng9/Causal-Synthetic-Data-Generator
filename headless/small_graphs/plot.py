"""Draw results.json as a PDF of cards: graph with colored cut X's, then three metric panels."""
import json, math, os, sys, textwrap
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results.json')))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'small_graphs.pdf')

INK, INK2, MUTED, LINE, PANEL, PAPER = '#16202b', '#465262', '#76818e', '#dce2e8', '#fbfcfd', '#ffffff'
EDGE, HIDDEN = '#3b4654', '#8b939d'
TINT = {'protected': '#f6dde8', 'outcome': '#f6ead2', 'admissible': '#d9efe5', 'other': '#eceff2', 'hidden': '#ffffff'}
RING = {'protected': '#b8396f', 'outcome': '#a86f00', 'admissible': '#11845d', 'other': '#6a7684', 'hidden': HIDDEN}
RULE = {'ftu': ('FTU', '#D55E00'), 'dp': ('DemPar', '#0072B2'), 'cf': ('CF', '#009E73')}
# Wider, lighter strokes underneath so identical lines stay visible as nested bands.
LW = {'cf': (3.4, .45), 'dp': (2.0, .9), 'ftu': (0.9, 1.0)}
MET = [('fairness gap (pp)', 0), ('S inference (skill)', 1), ('utility (skill)', 2)]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6, 'axes.linewidth': .4, 'pdf.fonttype': 42})

PW, PH, MX, MY, COLS, GAP = 17.0, 11.0, 0.45, 0.4, 6, 0.14
CW = (PW - 2 * MX - (COLS - 1) * GAP) / COLS
CH, SECH = 2.78, 0.42
U, PAD, R = 58, 22, 14
W, H = 3 * U + 2 * PAD, 2 * U + 2 * PAD

VALS = [[r['mean'][m] for s in D['sections'] for g in s['graphs'] for rr in g['res'].values() for r in rr] for m in range(3)]
GMAX = max(10, math.ceil(max(VALS[0]) / 10) * 10)
RANGE = [(0, GMAX)] + [(min(0, math.floor(min(v) * 10) / 10), math.ceil(max(v) * 10) / 10) for v in VALS[1:]]


def lab(name):
    return f'$\\mathregular{{{name[0]}_{{{name[1:]}}}}}$' if len(name) > 1 and name[1:].isdigit() else name


def draw_graph(ax, g):
    nodes = {n['id']: n for n in g['nodes']}
    mx, my = max(n['gx'] for n in g['nodes']), max(n['gy'] for n in g['nodes'])
    ux = 3 * U / mx if mx else U          # stretch narrow graphs across the card; relative layout unchanged
    ox, oy = PAD + (3 * U - mx * ux) / 2, PAD + (2 - my) * U / 2
    P = {i: (ox + n['gx'] * ux, oy + n['gy'] * U) for i, n in nodes.items()}
    cutby = {}
    for r in ('ftu', 'dp', 'cf'):
        for eid in g['cuts'][r]:
            cutby.setdefault(eid, []).append(r)
    for e in g['edges']:
        (ax_, ay), (bx, by) = P[e['a']], P[e['b']]
        dx, dy = bx - ax_, by - ay
        L = math.hypot(dx, dy)
        cx, cy = (ax_ + bx) / 2, (ay + by) / 2
        blocked = False
        for i, (px, py) in P.items():
            if i in (e['a'], e['b']):
                continue
            t = ((px - ax_) * dx + (py - ay) * dy) / (L * L)
            if 0 < t < 1 and math.hypot(ax_ + t * dx - px, ay + t * dy - py) < R + 4:
                blocked = True
        if blocked:
            cx, cy = cx - dy * 0.28, cy + dx * 0.28
        u1, u2 = math.hypot(cx - ax_, cy - ay), math.hypot(bx - cx, by - cy)
        sx, sy = ax_ + (cx - ax_) / u1 * R, ay + (cy - ay) / u1 * R
        ux, uy = (bx - cx) / u2, (by - cy) / u2
        tx, ty = bx - ux * (R + 1.5), by - uy * (R + 1.5)
        lx, ly = tx - ux * 6, ty - uy * 6
        hid = nodes[e['a']]['role'] == 'hidden' or nodes[e['b']]['role'] == 'hidden'
        col = HIDDEN if hid else EDGE
        ax.add_patch(PathPatch(Path([(sx, sy), (cx, cy), (lx, ly)], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                               fill=False, ec=col, lw=1.0, ls=(0, (1.5, 1.5)) if hid else '-', capstyle='round', zorder=2))
        ax.add_patch(Polygon([(tx, ty), (lx - uy * 4.2, ly + ux * 4.2), (lx + uy * 4.2, ly - ux * 4.2)], closed=True, fc=col, ec='none', zorder=2))
        rules = cutby.get(e['id'], [])
        if rules:
            ts = [i / 200 for i in range(201)]
            pts = [((1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t * t * lx, (1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t * t * ly) for t in ts]
            cum = [0.0]
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                cum.append(cum[-1] + math.hypot(x2 - x1, y2 - y1))
            Ltot = cum[-1]
            sp = min(13.0, (Ltot - 4) / max(1, len(rules)))
            hx = min(5.2, sp * 0.4)
            for j, r in enumerate(rules):
                d = Ltot / 2 + (j - (len(rules) - 1) / 2) * sp
                i = min(range(len(cum)), key=lambda i: abs(cum[i] - d))
                qx, qy = pts[i]
                for s in (1, -1):
                    ax.plot([qx - hx, qx + hx], [qy - s * hx, qy + s * hx], color=RULE[r][1], lw=2.3 if hx > 4 else 1.9, solid_capstyle='round', zorder=4,
                            path_effects=[pe.Stroke(linewidth=3.8 if hx > 4 else 3.2, foreground=PAPER), pe.Normal()])
    for i, n in nodes.items():
        x, y = P[i]
        hid = n['role'] == 'hidden'
        ax.add_patch(Circle((x, y), R, fc=TINT[n['role']], ec=RING[n['role']], lw=1.1, ls=(0, (2, 1.6)) if hid else '-', zorder=3))
        ax.text(x, y, lab(n['name']), ha='center', va='center', fontsize=6.6, color=MUTED if hid else INK, zorder=5, fontweight='bold')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_metrics(fig, x0, y0, w, h, g):
    pw, pg = (w - 2 * 0.12) / 3, 0.12
    xs = D['strengths']
    for k, (title, m) in enumerate(MET):
        ax = fig.add_axes([(x0 + k * (pw + pg)) / PW, y0 / PH, pw / PW, h / PH])
        lo, hi = RANGE[m]
        for r in ('cf', 'dp', 'ftu'):
            ys = [row['mean'][m] for row in g['res'][r]]
            ax.plot(xs, ys, color=RULE[r][1], lw=LW[r][0], alpha=LW[r][1], solid_capstyle='round', zorder=3)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '', '1'])
        ax.set_yticks([lo, (lo + hi) / 2, hi])
        ax.set_yticklabels([f'{lo:g}', '', f'{hi:g}'])
        if lo < 0:
            ax.axhline(0, color='#c3cbd5', lw=.5, zorder=1)
        ax.tick_params(length=1.5, width=.4, pad=1, labelsize=4.6, colors=MUTED)
        ax.grid(axis='y', color=LINE, lw=.4)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#c3cbd5')
        ax.set_title(title, fontsize=4.9, color=INK2, pad=1.5)
        ax.set_facecolor(PANEL)


def card(fig, x, ytop, g):
    y = ytop - CH
    fig.patches.append(FancyBboxPatch((x / PW, y / PH), CW / PW, CH / PH, boxstyle='round,pad=0,rounding_size=0.004',
                                      transform=fig.transFigure, fc=PANEL, ec=LINE, lw=.6, zorder=-5))
    fig.text((x + 0.08) / PW, (ytop - 0.08) / PH, f"{g['k']}", fontsize=5.5, color=MUTED, va='top', family='DejaVu Sans Mono')
    fig.text((x + 0.27) / PW, (ytop - 0.075) / PH, g['t'], fontsize=6.9, color=INK, va='top', fontweight='bold')
    gh = 1.22
    gw = gh * W / H
    ax = fig.add_axes([(x + (CW - gw) / 2) / PW, (ytop - 0.24 - gh) / PH, gw / PW, gh / PH])
    draw_graph(ax, g)
    fig.text((x + 0.08) / PW, (ytop - 0.27 - gh) / PH, textwrap.fill(g['note'], 62), fontsize=5.3, color=INK2, va='top', linespacing=1.3)
    draw_metrics(fig, x + 0.3, y + 0.13, CW - 0.4, 0.74, g)


def header(fig, ytop):
    fig.text(MX / PW, ytop / PH, 'Small Causal Graphs', fontsize=17, fontweight='bold', color=INK, va='top')
    fig.text(MX / PW, (ytop - 0.36) / PH,
             f"46 graphs of 2 to 6 nodes. The world is the graph with every edge intact. The release is the same graph sampled with a rule's edges cut "
             f"on a share of rows (the cut strength, 0 to 1); a cut parent is shuffled, as DECAF does. Every edge explains a {D['edge']:g} share of its child's latent variance,\n"
             f"lowered so no child's parents add to more than 0.9. S and Y are binary, 50/50; every other column, U included, is continuous and normal. "
             f"Hidden U is in the world but never in the release, and cuts see only the visible columns. {D['rows']:,} rows, mean of {len(D['seeds'])} seeds.",
             fontsize=7, color=INK2, va='top', linespacing=1.45)
    yy = ytop - 0.95
    x = MX
    for r in ('ftu', 'dp', 'cf'):
        name, col = RULE[r]
        fig.text(x / PW, yy / PH, '×', fontsize=15, color=col, fontweight='bold', va='center')
        fig.text((x + 0.2) / PW, yy / PH, name, fontsize=7.5, color=INK, fontweight='bold', va='center')
        x += 0.85
    fig.text(x / PW, yy / PH,
             "Cut marks sit in the order FTU, DemPar, CF along each edge. FTU cuts S→Y. CF cuts the non-admissible edges at Y that are tied to S given A. "
             "DemPar is CF with no admissible columns.",
             fontsize=6.6, color=INK2, va='center')
    yy -= 0.27
    fig.text(MX / PW, yy / PH,
             "Panels, x = cut strength from 0 to 1. A recipient fits logistic regression for Y on the release (every released column, S included) and is scored on a fresh table from the world.\n"
             f"Fairness gap: difference in mean predicted P(Y=1) between the S groups on world data, in percentage points (axis 0 to {GMAX}). "
             f"S inference: Brier skill of an attacker fit on the release to guess S from the other released columns, scored on world data (0 = no better than the base rate; axis {RANGE[1][0]:g} to {RANGE[1][1]:g}).\n"
             f"Utility: Brier skill of the recipient's model for Y on world data (axis {RANGE[2][0]:g} to {RANGE[2][1]:g}). Lines use the rule colors; wider, lighter lines sit underneath, so rules that cut the same edges show as nested bands. Every line starts at the same point, strength 0.",
             fontsize=6.6, color=INK2, va='top', linespacing=1.5)
    return ytop - 1.72


def build():
    pdf = PdfPages(OUT)
    fig, ytop, first = None, None, True

    def newpage():
        nonlocal fig, ytop
        if fig is not None:
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(PW, PH))
        fig.patch.set_facecolor(PAPER)
        ytop = PH - MY

    newpage()
    ytop = header(fig, ytop)
    for sec in D['sections']:
        gs = sec['graphs']
        rows = [gs[i:i + COLS] for i in range(0, len(gs), COLS)]
        for ri, row in enumerate(rows):
            need = (SECH if ri == 0 else 0) + CH
            if ytop - need < MY:
                newpage()
                if ri > 0:
                    fig.text(MX / PW, ytop / PH, sec['title'] + ' (continued)', fontsize=10.5, fontweight='bold', color=INK, va='top')
                    ytop -= SECH
            if ri == 0:
                fig.text(MX / PW, ytop / PH, sec['title'], fontsize=10.5, fontweight='bold', color=INK, va='top')
                fig.text(MX / PW, (ytop - 0.2) / PH, sec['sub'], fontsize=6.6, color=MUTED, va='top')
                ytop -= SECH
            for j, g in enumerate(row):
                card(fig, MX + j * (CW + GAP), ytop, g)
            ytop -= CH + GAP
        ytop -= 0.06
    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()


build()
