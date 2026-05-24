"""
Create Figure 1 for the IBERAMIA paper:
Timeformer ablation chain plus the token@time traceability idea.

The script writes:
  - outputs/figures/figure1_ablation_chain.html
  - outputs/figures/figure1_ablation_chain.json

The PNG is rendered from the self-contained HTML with Chrome headless, matching
the Figure 3 workflow used in this environment.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


OUT_DIR = Path("outputs/figures")
HTML_PATH = OUT_DIR / "figure1_ablation_chain.html"
JSON_PATH = OUT_DIR / "figure1_ablation_chain.json"


COLORS = {
    "ink": "#1c3557",
    "muted": "#60708b",
    "grid": "#dce6f2",
    "panel": "#f7f9fc",
    "static": "#636EFA",
    "time": "#EF553B",
    "token": "#00CC96",
    "memory": "#AB63FA",
    "context_a": "#2f80ed",
    "context_b": "#e05252",
}


def add_box(
    fig: go.Figure,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    title: str,
    body: str,
    color: str,
) -> None:
    fig.add_shape(
        type="rect",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        line={"color": color, "width": 2},
        fillcolor="white",
        layer="below",
    )
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=y1 - 0.045,
        text=f"<b>{title}</b>",
        showarrow=False,
        font={"size": 15, "color": color},
        align="center",
    )
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=(y0 + y1) / 2 - 0.015,
        text=body,
        showarrow=False,
        font={"size": 12, "color": COLORS["ink"]},
        align="center",
    )


def add_arrow(fig: go.Figure, x0: float, y0: float, x1: float, y1: float) -> None:
    fig.add_annotation(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=1.7,
        arrowcolor=COLORS["muted"],
    )


def build_figure() -> go.Figure:
    fig = go.Figure()

    # Background panels.
    for x0, x1, title in [
        (0.02, 0.30, "token@time interface"),
        (0.33, 0.98, "Timeformer ablation chain"),
    ]:
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=0.12,
            x1=x1,
            y1=0.88,
            line={"color": COLORS["grid"], "width": 1},
            fillcolor=COLORS["panel"],
            layer="below",
        )
        fig.add_annotation(
            x=x0 + 0.015,
            y=0.84,
            text=f"<b>{title}</b>",
            showarrow=False,
            xanchor="left",
            font={"size": 15, "color": COLORS["ink"]},
        )

    # Panel A: token@time concept.
    fig.add_shape(
        type="circle",
        x0=0.065,
        y0=0.56,
        x1=0.145,
        y1=0.70,
        line={"color": COLORS["context_a"], "width": 2},
        fillcolor="#e8f1ff",
    )
    fig.add_shape(
        type="circle",
        x0=0.175,
        y0=0.30,
        x1=0.255,
        y1=0.44,
        line={"color": COLORS["context_b"], "width": 2},
        fillcolor="#fff0f0",
    )
    fig.add_annotation(
        x=0.105,
        y=0.63,
        text="<b>S11@t2</b><br>near context A",
        showarrow=False,
        font={"size": 12, "color": COLORS["context_a"]},
    )
    fig.add_annotation(
        x=0.215,
        y=0.37,
        text="<b>S11@t8</b><br>near context B",
        showarrow=False,
        font={"size": 12, "color": COLORS["context_b"]},
    )
    add_arrow(fig, 0.135, 0.56, 0.19, 0.45)
    fig.add_annotation(
        x=0.16,
        y=0.515,
        text="same surface token<br>different epoch",
        showarrow=False,
        font={"size": 11, "color": COLORS["muted"]},
    )

    fig.add_annotation(
        x=0.16,
        y=0.20,
        text="Traceability = query and compare<br>representations as token@time",
        showarrow=False,
        font={"size": 12, "color": COLORS["ink"]},
        align="center",
    )

    # Panel B: ablation chain.
    y0, y1 = 0.38, 0.68
    boxes = [
        (
            0.36,
            0.49,
            "Static",
            "Token + position<br><span style='color:#60708b'>no epoch signal</span>",
            COLORS["static"],
        ),
        (
            0.515,
            0.645,
            "Additive Time",
            "Token + position<br>+ TimeEncoding(t)",
            COLORS["time"],
        ),
        (
            0.67,
            0.80,
            "Token-Time",
            "f(Token,<br>TimeEncoding(t))",
            COLORS["token"],
        ),
        (
            0.825,
            0.955,
            "Memory-Aug.",
            "Token-Time<br>+ m(S,t&lt;k)",
            COLORS["memory"],
        ),
    ]
    for x0, x1, title, body, color in boxes:
        add_box(fig, x0, y0, x1, y1, title, body, color)

    for left, right in zip(boxes, boxes[1:]):
        add_arrow(fig, left[1] + 0.01, 0.53, right[0] - 0.01, 0.53)
        fig.add_annotation(
            x=(left[1] + right[0]) / 2,
            y=0.53,
            text="<b>&gt;</b>",
            showarrow=False,
            font={"size": 24, "color": COLORS["muted"]},
        )

    # Mechanism labels under the chain.
    labels = [
        (0.425, "baseline"),
        (0.58, "global epoch<br>conditioning"),
        (0.735, "token-specific<br>conditioning"),
        (0.89, "causal historical<br>prototype attention"),
    ]
    for x, text in labels:
        fig.add_annotation(
            x=x,
            y=0.245,
            text=text,
            showarrow=False,
            font={"size": 11, "color": COLORS["muted"]},
            align="center",
        )

    fig.add_annotation(
        x=0.66,
        y=0.80,
        text="<b>Increasing temporal capacity</b>",
        showarrow=False,
        font={"size": 13, "color": COLORS["ink"]},
    )
    add_arrow(fig, 0.52, 0.785, 0.80, 0.785)

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=520,
        margin={"l": 24, "r": 24, "t": 62, "b": 30},
        title={
            "text": "Figure 1. Timeformer ablation chain and token@time interface",
            "x": 0.02,
            "xanchor": "left",
        },
        font={"family": "Arial", "size": 13, "color": COLORS["ink"]},
        xaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        yaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.write_html(HTML_PATH, include_plotlyjs=True, full_html=True)
    fig.write_json(JSON_PATH)
    print(f"Wrote {HTML_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
