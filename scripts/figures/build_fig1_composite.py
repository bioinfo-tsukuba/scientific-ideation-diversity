"""Composite Fig 1: side-by-side concat of effort-axis Fig 1 + prompt-axis Pareto Fig.

Concats two existing per-figure PDFs at the page-image level (no matplotlib
re-render), preserving each figure's original layout and (a)/(b) panel labels
that are already baked in by the source scripts.

Inputs (must already be built by their respective scripts):
- outputs/figures/pair_distance_vs_reasoning_tokens.pdf    (Panel A: effort axis)
- outputs/figures/prompt_sensitivity_token_pareto.pdf      (Panel B: prompt Pareto)

Output:
- outputs/figures/fig1_composite.pdf

Strategy: scale both panels to a common target height, place horizontally with
a small gap, leaving each panel's internal layout untouched.
"""

from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PANEL_A_PDF = FIG_DIR / "pair_distance_vs_reasoning_tokens.pdf"
PANEL_B_PDF = FIG_DIR / "prompt_sensitivity_token_pareto.pdf"
OUT_PDF = FIG_DIR / "fig1_composite.pdf"

TARGET_HEIGHT_PT = 240.0
GAP_PT = 12.0


def main() -> None:
    page_a = PdfReader(PANEL_A_PDF).pages[0]
    page_b = PdfReader(PANEL_B_PDF).pages[0]

    w_a, h_a = float(page_a.mediabox.width), float(page_a.mediabox.height)
    w_b, h_b = float(page_b.mediabox.width), float(page_b.mediabox.height)

    sx_a = sy_a = TARGET_HEIGHT_PT / h_a
    sx_b = sy_b = TARGET_HEIGHT_PT / h_b
    new_w_a = w_a * sx_a
    new_w_b = w_b * sx_b

    total_w = new_w_a + GAP_PT + new_w_b
    total_h = TARGET_HEIGHT_PT

    new_page = PageObject.create_blank_page(width=total_w, height=total_h)

    tf_a = Transformation().scale(sx_a, sy_a)
    new_page.merge_transformed_page(page_a, tf_a)

    tf_b = Transformation().scale(sx_b, sy_b).translate(new_w_a + GAP_PT, 0)
    new_page.merge_transformed_page(page_b, tf_b)

    writer = PdfWriter()
    writer.add_page(new_page)
    with OUT_PDF.open("wb") as f:
        writer.write(f)
    print(
        f"saved {OUT_PDF.relative_to(REPO_ROOT)}: "
        f"{total_w:.1f}pt x {total_h:.1f}pt "
        f"(Panel A {new_w_a:.1f}pt + gap {GAP_PT:.0f}pt + Panel B {new_w_b:.1f}pt)"
    )


if __name__ == "__main__":
    main()
