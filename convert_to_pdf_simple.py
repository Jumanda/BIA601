#!/usr/bin/env python3
"""
Convert Markdown to PDF using simple method
"""
import sys
from pathlib import Path
import markdown
from xhtml2pdf import pisa

def markdown_to_pdf(input_file, output_file):
    """Convert Markdown to PDF"""
    
    # Read markdown file
    md_content = Path(input_file).read_text(encoding='utf-8')
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['codehilite', 'fenced_code'])
    
    # Add RTL direction and styling
    full_html = f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: Arial, sans-serif;
                direction: rtl;
                text-align: right;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #2c3e50;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }}
            h1 {{
                font-size: 2em;
                border-bottom: 3px solid #3498db;
                padding-bottom: 0.3em;
            }}
            h2 {{
                font-size: 1.5em;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 0.2em;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                padding: 15px;
                overflow-x: auto;
                direction: ltr;
                text-align: left;
            }}
            ul, ol {{
                padding-right: 2em;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert to PDF
    with open(output_file, 'wb') as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file, encoding='utf-8')
    
    if pisa_status.err:
        print(f"❌ خطأ في التحويل: {pisa_status.err}")
        sys.exit(1)
    else:
        print(f"✅ تم تحويل الملف بنجاح!")
        print(f"📄 الملف الناتج: {output_file}")

if __name__ == '__main__':
    input_file = 'REPORT_PCA_PERSONAL.md'
    output_file = 'REPORT_PCA_PERSONAL.pdf'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"❌ الملف غير موجود: {input_file}")
        sys.exit(1)
    
    markdown_to_pdf(input_file, output_file)


