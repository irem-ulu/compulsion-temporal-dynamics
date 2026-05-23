"""Render the data-generating DAG using matplotlib only — no graphviz dep.

Reads :data:`ctd.data.generate.CAUSAL_DAG` so the picture is locked to
the simulator. If you change the DGP, the figure updates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from ..data.generate import CAUSAL_DAG

# Hand-laid coordinates — keep nodes from overlapping and group the
# person-level latents on the left, observed time-varying outcomes on
# the right.
LAYOUT: dict[str, tuple[float, float]] = {
    "PersonStress": (0.0, 2.0),
    "PersonOutside": (0.0, 1.0),
    "PersonCompulsion": (0.0, 0.0),
    "circadian": (1.5, 3.0),
    "Stress_{t-1}": (1.5, 0.5),
    "Stress": (3.0, 2.0),
    "OutsideTime": (3.0, 1.0),
    "Compulsions": (3.0, 0.0),
}


def render_dag(out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(-0.7, 4.0)
    ax.set_ylim(-0.6, 3.6)
    ax.axis("off")

    node_style = dict(boxstyle="round,pad=0.35", ec="black", lw=1)
    fill = {
        "Stress": "#dbeafe",
        "OutsideTime": "#dbeafe",
        "Compulsions": "#dbeafe",
        "circadian": "#fef3c7",
        "Stress_{t-1}": "#fef3c7",
        "PersonStress": "#fce7f3",
        "PersonOutside": "#fce7f3",
        "PersonCompulsion": "#fce7f3",
    }

    for label, (x, y) in LAYOUT.items():
        ax.text(x, y, label, ha="center", va="center",
                bbox={**node_style, "fc": fill.get(label, "white")})

    for src, dst, sign in CAUSAL_DAG:
        if src not in LAYOUT or dst not in LAYOUT:
            continue
        x1, y1 = LAYOUT[src]
        x2, y2 = LAYOUT[dst]
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->", mutation_scale=14,
            color="C2" if sign == "+" else "C3",
            lw=1.4,
            shrinkA=22, shrinkB=22,
            connectionstyle="arc3,rad=0.08",
        )
        ax.add_patch(arrow)

    ax.set_title("Data-generating DAG (green = positive, red = negative)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
