import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf_report(
    output_path: str,
    dataset_path: str,
    target_column: str,
    profile: dict,
    strategy: dict,
    execution_status: bool,
    retries_used: int
) -> str:
    """Generates an executive PDF summary report of the autonomous MLOps run."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = styles['BodyText']
    
    story = []
    
    # Title
    story.append(Paragraph("Autonomous Agentic MLOps — Run Summary", title_style))
    story.append(Spacer(1, 8))
    
    # Metadata Table
    status_text = "SUCCESS ✅" if execution_status else "FAILED ❌"
    summary_data = [
        ["Dataset Target", f"{dataset_path} (Target: {target_column})"],
        ["Pipeline Status", status_text],
        ["Debugger Retries", str(retries_used)]
    ]
    t = Table(summary_data, colWidths=[140, 380])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))
    
    # Section 1: Profiling
    story.append(Paragraph("1. Dataset Profiling Summary", section_style))
    shape = profile.get("dataset_shape", {})
    col_count = len(profile.get("columns", {}))
    story.append(Paragraph(f"• <b>Dimensions:</b> {shape.get('rows', 'N/A')} rows, {shape.get('columns', col_count)} columns", body_style))
    story.append(Spacer(1, 10))
    
    # Section 2: Strategy Strategy
    story.append(Paragraph("2. LLM Transformation Strategy", section_style))
    strat = strategy.get("data_strategy", {})
    
    for category, header in [("data_cleaning", "Data Cleaning"), ("feature_engineering", "Feature Engineering"), ("model_selection", "Model Selection")]:
        steps = strat.get(category, {}).get("steps", [])
        if steps:
            story.append(Paragraph(f"<b>{header}:</b>", body_style))
            for s in steps:
                step_name = s.get("step", "")
                desc = s.get("description", "")
                cols = ", ".join(s.get("column_names", []))
                story.append(Paragraph(f"• <i>{step_name}</i> ({cols}): {desc}", body_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    return output_path