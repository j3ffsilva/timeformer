"""
Create Figure 2 for the IBERAMIA paper:
Controlled design comparison — 4-column layout showing
Standard → Additive → Joint → Timeformer.

Changes from v1:
  - Title updated to "Controlled design comparison"
  - Formula subtitles added below each column header
  - Memory + Gate row moved between Token-Time Interaction and Encoder
    (architecturally correct: memory modifies x_s *before* the encoder)

The script writes:
  - outputs/figures/figure2_architecture_detail.html
  - outputs/figures/figure2_architecture_detail.json

The PNG is rendered from the HTML with Chrome headless (see compile.sh).
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

OUT_DIR   = Path("outputs/figures")
HTML_PATH = OUT_DIR / "figure2_architecture_detail.html"
JSON_PATH = OUT_DIR / "figure2_architecture_detail.json"

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "ink":         "#1c3557",
    "muted":       "#8899aa",
    "grid":        "#dce6f2",
    "panel":       "#f7f9fc",
    "token":       "#636EFA",
    "time":        "#EF553B",
    "interaction": "#00CC96",
    "encoder":     "#2f80ed",
    "memory":      "#AB63FA",
    "output":      "#e05252",
    "new_fill":    "#fffbe6",
    "base_fill":   "#ffffff",
    "absent_line": "#cccccc",
    "absent_text": "#cccccc",
}

# ── Layout constants ──────────────────────────────────────────────────────────
COL_CENTERS = [0.125, 0.375, 0.625, 0.875]
COL_W  = 0.11
BOX_H  = 0.048

HEADER_Y  = 0.955
FORMULA_Y = 0.916

# Six rows, top to bottom: token → time → interact → mem_attn → encoder → output
ROWS = {
    "token":    0.790,
    "time":     0.655,
    "interact": 0.520,
    "mem_attn": 0.375,   # Memory + Gate — between interact and encoder
    "encoder":  0.225,
    "output":   0.085,
}

# Timeformer sub-boxes within the mem_attn row
HALF_W_TF  = 0.042
MEM_OFFSET = COL_W * 0.44

VARIANT_LABELS = [
    "Standard<br>Transformer",
    "Additive<br>Time-Conditioned",
    "Token-Time<br>(Joint)",
    "Timeformer<br>(Full Model)",
]

FORMULAS = [
    "x<sub>i</sub> = e(w<sub>i</sub>) + p<sub>i</sub>",
    "x<sub>i</sub> = e(w<sub>i</sub>) + p<sub>i</sub> + τ(t)",
    "x<sub>i</sub> = W[e(w<sub>i</sub>); τ(t)] + p<sub>i</sub>",
    "x̃<sub>s</sub> = x<sub>s</sub> + g · Attn(x<sub>s</sub>, M<sub>s</sub>)",
]

HEADER_COLORS = [C["muted"], C["time"], C["interaction"], C["memory"]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def y_top(row: str) -> float:
    return ROWS[row] + BOX_H

def y_bot(row: str) -> float:
    return ROWS[row] - BOX_H


def box(fig: go.Figure,
        cx: float, cy: float,
        title: str, body: str,
        line_color: str,
        fill: str | None = None,
        absent: bool = False) -> None:
    if fill is None:
        fill = C["base_fill"]
    lc = C["absent_line"] if absent else line_color
    fc = "#f8f8f8"        if absent else fill
    fig.add_shape(type="rect",
                  x0=cx - COL_W, y0=cy - BOX_H,
                  x1=cx + COL_W, y1=cy + BOX_H,
                  line={"color": lc, "width": 1.5 if absent else 2,
                        "dash": "dot" if absent else "solid"},
                  fillcolor=fc, layer="below")
    tc = C["absent_text"] if absent else line_color
    fig.add_annotation(x=cx, y=cy + BOX_H - 0.018,
                       text=f"<b>{title}</b>",
                       showarrow=False,
                       font={"size": 10, "color": tc}, align="center")
    bc = C["absent_text"] if absent else C["ink"]
    fig.add_annotation(x=cx, y=cy - 0.014,
                       text=body,
                       showarrow=False,
                       font={"size": 9, "color": bc}, align="center")


def arrow(fig: go.Figure,
          x0: float, y0: float,
          x1: float, y1: float,
          color: str = C["muted"]) -> None:
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                       xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3,
                       arrowsize=1.0, arrowwidth=1.8,
                       arrowcolor=color)


# ── Figure builder ────────────────────────────────────────────────────────────

def build_figure() -> go.Figure:
    fig = go.Figure()

    # Background panel
    fig.add_shape(type="rect", x0=0.01, y0=0.01, x1=0.99, y1=0.99,
                  line={"color": C["grid"], "width": 1},
                  fillcolor=C["panel"], layer="below")

    # Column dividers
    for i in range(1, 4):
        xd = (COL_CENTERS[i - 1] + COL_CENTERS[i]) / 2
        fig.add_shape(type="line",
                      x0=xd, y0=0.04, x1=xd, y1=0.935,
                      line={"color": C["grid"], "width": 1, "dash": "dot"})

    # ── Column headers and formula subtitles ──────────────────────────────────
    for cx, label, formula, hc in zip(
            COL_CENTERS, VARIANT_LABELS, FORMULAS, HEADER_COLORS):
        fig.add_annotation(x=cx, y=HEADER_Y,
                           text=f"<b>{label}</b>",
                           showarrow=False,
                           font={"size": 11, "color": hc}, align="center")
        fig.add_annotation(x=cx, y=FORMULA_Y,
                           text=formula,
                           showarrow=False,
                           font={"size": 9, "color": hc}, align="center")

    # ── Row: Token input (all variants) ──────────────────────────────────────
    for cx in COL_CENTERS:
        box(fig, cx, ROWS["token"], "Token input", "S V O + position", C["token"])

    # ── Row: Time input ───────────────────────────────────────────────────────
    box(fig, COL_CENTERS[0], ROWS["time"],
        "Time input", "— absent —", C["time"], absent=True)
    box(fig, COL_CENTERS[1], ROWS["time"],
        "Time input", "τ(t)  continuous", C["time"], fill=C["new_fill"])
    box(fig, COL_CENTERS[2], ROWS["time"],
        "Time input", "τ(t)  continuous", C["time"])
    box(fig, COL_CENTERS[3], ROWS["time"],
        "Time input", "τ(t)  continuous", C["time"])

    # ── Row: Token-Time Interaction ───────────────────────────────────────────
    box(fig, COL_CENTERS[0], ROWS["interact"],
        "Token-Time<br>Interaction", "— absent —", C["interaction"], absent=True)
    box(fig, COL_CENTERS[1], ROWS["interact"],
        "Token-Time<br>Interaction", "— absent —", C["interaction"], absent=True)
    box(fig, COL_CENTERS[2], ROWS["interact"],
        "Token-Time<br>Interaction", "W [e(w); τ(t)]", C["interaction"],
        fill=C["new_fill"])
    box(fig, COL_CENTERS[3], ROWS["interact"],
        "Token-Time<br>Interaction", "W [e(w); τ(t)]", C["interaction"])

    # ── Row: Memory + Gate ────────────────────────────────────────────────────
    # Standard, Additive, Joint: absent
    for cx in COL_CENTERS[:3]:
        box(fig, cx, ROWS["mem_attn"],
            "Memory + Gate", "— absent —", C["memory"], absent=True)

    # Timeformer: two sub-boxes side by side
    mem_cx  = COL_CENTERS[3] - MEM_OFFSET
    attn_cx = COL_CENTERS[3] + MEM_OFFSET
    y0_mem  = ROWS["mem_attn"] - BOX_H
    y1_mem  = ROWS["mem_attn"] + BOX_H

    for cx_sub, title_sub, body_sub in [
        (mem_cx,  "Prototype<br>Memory",  "m(s,t),  t &lt; t_k"),
        (attn_cx, "Temporal<br>Attention", "gate  g"),
    ]:
        fig.add_shape(type="rect",
                      x0=cx_sub - HALF_W_TF, y0=y0_mem,
                      x1=cx_sub + HALF_W_TF, y1=y1_mem,
                      line={"color": C["memory"], "width": 2},
                      fillcolor=C["new_fill"], layer="below")
        fig.add_annotation(x=cx_sub, y=y1_mem - 0.020,
                           text=f"<b>{title_sub}</b>",
                           showarrow=False,
                           font={"size": 9, "color": C["memory"]}, align="center")
        fig.add_annotation(x=cx_sub, y=y0_mem + 0.018,
                           text=body_sub,
                           showarrow=False,
                           font={"size": 8, "color": C["ink"]}, align="center")

    # ── Row: Transformer Encoder (all variants) ───────────────────────────────
    for cx in COL_CENTERS:
        box(fig, cx, ROWS["encoder"],
            "Transformer<br>Encoder", "contextualizes sentence", C["encoder"])

    # ── Row: h_s output (all variants) ───────────────────────────────────────
    for cx in COL_CENTERS:
        box(fig, cx, ROWS["output"],
            "h<sub>s</sub>", "token@time state", C["output"])

    # ── Arrows ────────────────────────────────────────────────────────────────

    # Standard: Token → Encoder (skips time, interact, mem_attn) → output
    arrow(fig, COL_CENTERS[0], y_bot("token"),
               COL_CENTERS[0], y_top("encoder"), C["token"])
    arrow(fig, COL_CENTERS[0], y_bot("encoder"),
               COL_CENTERS[0], y_top("output"),  C["encoder"])

    # Additive: Token → Encoder, Time → Encoder (skips interact, mem_attn) → output
    arrow(fig, COL_CENTERS[1], y_bot("token"),
               COL_CENTERS[1], y_top("encoder"), C["token"])
    arrow(fig, COL_CENTERS[1], y_bot("time"),
               COL_CENTERS[1], y_top("encoder"), C["time"])
    fig.add_annotation(
        x=COL_CENTERS[1] + 0.022,
        y=(y_bot("time") + y_top("encoder")) / 2,
        text="<i>add</i>", showarrow=False,
        font={"size": 9, "color": C["time"]})
    arrow(fig, COL_CENTERS[1], y_bot("encoder"),
               COL_CENTERS[1], y_top("output"),  C["encoder"])

    # Joint: Token → Interact, Time → Interact, Interact → Encoder (skips mem_attn) → output
    arrow(fig, COL_CENTERS[2], y_bot("token"),
               COL_CENTERS[2], y_top("interact"),  C["token"])
    arrow(fig, COL_CENTERS[2], y_bot("time"),
               COL_CENTERS[2], y_top("interact"),  C["time"])
    arrow(fig, COL_CENTERS[2], y_bot("interact"),
               COL_CENTERS[2], y_top("encoder"),   C["interaction"])
    arrow(fig, COL_CENTERS[2], y_bot("encoder"),
               COL_CENTERS[2], y_top("output"),    C["encoder"])

    # Timeformer: Token → Interact, Time → Interact,
    #             Interact → mem_attn, (Memory → Attention horizontal),
    #             mem_attn → Encoder → output
    arrow(fig, COL_CENTERS[3], y_bot("token"),
               COL_CENTERS[3], y_top("interact"),   C["token"])
    arrow(fig, COL_CENTERS[3], y_bot("time"),
               COL_CENTERS[3], y_top("interact"),   C["time"])
    arrow(fig, COL_CENTERS[3], y_bot("interact"),
               COL_CENTERS[3], y_top("mem_attn"),   C["interaction"])
    # Memory → Temporal Attention (horizontal within mem_attn row)
    arrow(fig, mem_cx  + HALF_W_TF, ROWS["mem_attn"],
               attn_cx - HALF_W_TF, ROWS["mem_attn"], C["memory"])
    # mem_attn row → Encoder
    arrow(fig, COL_CENTERS[3], y_bot("mem_attn"),
               COL_CENTERS[3], y_top("encoder"),    C["memory"])
    # Encoder → output
    arrow(fig, COL_CENTERS[3], y_bot("encoder"),
               COL_CENTERS[3], y_top("output"),     C["encoder"])

    # ── Legend ────────────────────────────────────────────────────────────────
    fig.add_shape(type="rect", x0=0.03, y0=0.020, x1=0.080, y1=0.048,
                  line={"color": C["interaction"], "width": 1.5},
                  fillcolor=C["new_fill"], layer="below")
    fig.add_annotation(x=0.087, y=0.034,
                       text="new component at this step",
                       showarrow=False,
                       font={"size": 9, "color": C["muted"]},
                       xanchor="left")

    fig.add_shape(type="rect", x0=0.43, y0=0.020, x1=0.480, y1=0.048,
                  line={"color": C["absent_line"], "width": 1.5, "dash": "dot"},
                  fillcolor="#f8f8f8", layer="below")
    fig.add_annotation(x=0.487, y=0.034,
                       text="component absent in this variant",
                       showarrow=False,
                       font={"size": 9, "color": C["muted"]},
                       xanchor="left")

    fig.update_layout(
        template="plotly_white",
        width=1400,
        height=780,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        title={
            "text": "Figure 2. Controlled design comparison",
            "x": 0.02, "xanchor": "left",
        },
        font={"family": "Arial", "size": 12, "color": C["ink"]},
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
