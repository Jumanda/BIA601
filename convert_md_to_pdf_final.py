#!/usr/bin/env python3
"""
Convert Markdown to PDF using reportlab with RTL support
"""
from pathlib import Path
import markdown
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from arabic_reshaper import reshape
from bidi.algorithm import get_display

def markdown_to_pdf(input_file, output_file):
    """Convert Markdown to PDF with RTL support"""
    
    # Read markdown
    md_content = Path(input_file).read_text(encoding='utf-8')
    
    # Convert to HTML first
    html_content = markdown.markdown(md_content, extensions=['fenced_code'])
    
    # Create PDF
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Create custom RTL style
    rtl_style = ParagraphStyle(
        'CustomRTL',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        alignment=2,  # Right align
        spaceAfter=12,
    )
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=24,
        alignment=2,
        spaceAfter=20,
        spaceBefore=20,
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=20,
        alignment=2,
        spaceAfter=15,
        spaceBefore=15,
    )
    
    # Parse HTML and create story
    story = []
    
    # Simple parser (you can improve this)
    lines = html_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.3*cm))
            continue
            
        # Handle headings
        if line.startswith('<h1>'):
            text = line.replace('<h1>', '').replace('</h1>', '')
            story.append(Paragraph(get_display(reshape(text)), title_style))
        elif line.startswith('<h2>'):
            text = line.replace('<h2>', '').replace('</h2>', '')
            story.append(Paragraph(get_display(reshape(text)), heading2_style))
        elif line.startswith('<p>'):
            text = line.replace('<p>', '').replace('</p>', '')
            text = text.replace('<strong>', '<b>').replace('</strong>', '</b>')
            story.append(Paragraph(get_display(reshape(text)), rtl_style))
        elif line.startswith('<pre>'):
            # Code blocks
            code_text = line.replace('<pre>', '').replace('</pre>', '').replace('<code>', '').replace('</code>', '')
            story.append(Preformatted(get_display(reshape(code_text)), rtl_style))
        else:
            # Regular text
            text = line.replace('<strong>', '<b>').replace('</strong>', '</b>')
            if text:
                story.append(Paragraph(get_display(reshape(text)), rtl_style))
    
    # Build PDF
    doc.build(story)
    print(f"✅ تم تحويل الملف بنجاح!")
    print(f"📄 الملف الناتج: {output_file}")

if __name__ == '__main__':
    markdown_to_pdf('REPORT_PCA_PERSONAL.md', 'REPORT_PCA_PERSONAL.pdf')


