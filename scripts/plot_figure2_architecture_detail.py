"""
Create Figure 2 for the IBERAMIA paper:
compact Timeformer architecture detail.

The script writes:
  - outputs/figures/figure2_architecture_detail.html
  - outputs/figures/figure2_architecture_detail.json

The PNG is rendered from the self-contained HTML with Chrome headless, matching
the Figure 1/Figure 3 workflow used in this environment.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


OUT_DIR = Path("outputs/figures")
HTML_PATH = OUT_DIR / "figure2_architecture_detail.html"
JSON_PATH = OUT_DIR / "figure2_architecture_detail.json"

COLORS = {
    "ink": "#1c3557",
    "muted": "#60708b",
    "grid": "#dce6f2",
    "panel": "#f7f9fc",
    "token": "#636EFA",
    "time": "#EF553B",
    "interaction": "#00CC96",
    "encoder": "#2f80ed",
    "memory": "#AB63FA",
    "output": "#e05252",
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
    fill: str = "white",
) -> None:
    fig.add_shape(
        type="rect",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        line={"color": color, "width": 2},
        fillcolor=fill,
        layer="below",
    )
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=y1 - 0.04,
        text=f"<b>{title}</b>",
        showarrow=False,
        font={"size": 14, "color": color},
        align="center",
    )
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=(y0 + y1) / 2 - 0.025,
        text=body,
        showarrow=False,
        font={"size": 12, "color": COLORS["ink"]},
        align="center",
    )


def add_arrow(
    fig: go.Figure,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = COLORS["muted"],
    width: float = 1.8,
) -> None:
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
        arrowsize=1.05,
        arrowwidth=width,
        arrowcolor=color,
    )


def build_figure() -> go.Figure:
    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=0.03,
        y0=0.10,
        x1=0.97,
        y1=0.90,
        line={"color": COLORS["grid"], "width": 1},
        fillcolor=COLORS["panel"],
        layer="below",
    )

    # Main flow boxes.
    add_box(
        fig,
        0.07,
        0.55,
        0.22,
        0.77,
        "Token input",
        "S V O<br>+ position",
        COLORS["token"],
    )
    add_box(
        fig,
        0.07,
        0.22,
        0.22,
        0.44,
        "Time input",
        "epoch t<br>continuous encoding",
        COLORS["time"],
    )
    add_box(
        fig,
        0.31,
        0.39,
        0.49,
        0.65,
        "Token-Time Interaction",
        "f(TokenEmb,<br>TimeEncoding(t))",
        COLORS["interaction"],
    )
    add_box(
        fig,
        0.57,
        0.39,
        0.73,
        0.65,
        "Transformer Encoder",
        "contextualizes<br>current sentence",
        COLORS["encoder"],
    )
    add_box(
        fig,
        0.81,
        0.39,
        0.94,
        0.65,
        "h(subject)",
        "queryable<br>token@time state",
        COLORS["output"],
    )

    # Memory module.
    add_box(
        fig,
        0.55,
        0.12,
        0.75,
        0.28,
        "PrototypeMemory",
        "m(S,t0)...m(S,t<k)",
        COLORS["memory"],
        fill="#fbf7ff",
    )
    add_box(
        fig,
        0.79,
        0.12,
        0.95,
        0.28,
        "Temporal attention",
        "updates subject<br>before encoder",
        COLORS["memory"],
        fill="#fbf7ff",
    )

    # Arrows.
    add_arrow(fig, 0.22, 0.66, 0.31, 0.55, COLORS["token"])
    add_arrow(fig, 0.22, 0.33, 0.31, 0.49, COLORS["time"])
    add_arrow(fig, 0.49, 0.52, 0.57, 0.52)
    add_arrow(fig, 0.73, 0.52, 0.81, 0.52)
    add_arrow(fig, 0.75, 0.20, 0.79, 0.20, COLORS["memory"])
    add_arrow(fig, 0.87, 0.28, 0.66, 0.39, COLORS["memory"])

    # Labels and equations.
    fig.add_annotation(
        x=0.40,
        y=0.75,
        text="Token-Time Transformer core",
        showarrow=False,
        font={"size": 13, "color": COLORS["interaction"]},
    )
    fig.add_shape(
        type="line",
        x0=0.305,
        y0=0.715,
        x1=0.735,
        y1=0.715,
        line={"color": COLORS["interaction"], "width": 2},
    )
    fig.add_annotation(
        x=0.75,
        y=0.79,
        text="Memory-Augmented Timeformer extension",
        showarrow=False,
        font={"size": 13, "color": COLORS["memory"]},
    )
    fig.add_shape(
        type="line",
        x0=0.55,
        y0=0.315,
        x1=0.95,
        y1=0.315,
        line={"color": COLORS["memory"], "width": 2, "dash": "dot"},
    )

    fig.add_annotation(
        x=0.50,
        y=0.16,
        text="causal: memory contains only t &lt; k",
        showarrow=False,
        font={"size": 11, "color": COLORS["muted"]},
        align="right",
    )

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=520,
        margin={"l": 24, "r": 24, "t": 64, "b": 30},
        title={
            "text": "Figure 2. Timeformer architecture detail",
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
