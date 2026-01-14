from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

@dataclass
class ReportMeta:
    title: str
    units_note: str
    project_note: str = ""
    page_size: str = "LETTER"  # "LETTER" or "A4"

def _pagesize(meta: ReportMeta):
    if meta.page_size.upper() == "A4":
        return A4
    return letter

def make_summary_pdf(
    meta: ReportMeta,
    summary_df,
    top_n: int = 30,
    element_plot_png: Optional[bytes] = None,
) -> bytes:
    """Create a compact PDF report and return bytes.

    summary_df: dataframe with controlling combo summary per element.
    element_plot_png: optional PNG bytes to embed (a single image).
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=_pagesize(meta),
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        title=meta.title,
        author="Practical Structural Analysis",
    )
    styles = getSampleStyleSheet()
    story: List = []

    story.append(Paragraph(meta.title, styles["Title"]))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"Units: {meta.units_note}", styles["Normal"]))
    if meta.project_note:
        story.append(Paragraph(meta.project_note, styles["Normal"]))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Governing combo summary (by element)", styles["Heading2"]))
    story.append(Paragraph(f"Showing top {top_n} rows (sorted by M_max_abs).", styles["Normal"]))
    story.append(Spacer(1, 0.1*inch))

    df = summary_df.copy()
    if len(df) > top_n:
        df = df.head(top_n)

    cols = list(df.columns)
    data = [cols] + df.astype(str).values.tolist()

    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2b2b2b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    story.append(tbl)

    if element_plot_png is not None:
        story.append(PageBreak())
        story.append(Paragraph("Selected element envelope plots", styles["Heading2"]))
        story.append(Spacer(1, 0.15*inch))
        img = Image(BytesIO(element_plot_png))
        img.drawWidth = 6.5*inch
        img.drawHeight = img.drawHeight * (img.drawWidth / img.drawWidth)  # keep aspect if possible
        story.append(img)

    doc.build(story)
    return buf.getvalue()
