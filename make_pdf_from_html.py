#!/usr/bin/env python3
"""
Create a properly styled HTML file that can be printed to PDF from browser
"""
from pathlib import Path
import markdown

# Read markdown
md_content = Path('REPORT_PCA_PERSONAL.md').read_text(encoding='utf-8')

# Convert to HTML
html_body = markdown.markdown(md_content, extensions=['codehilite', 'fenced_code'])

# Create full HTML with RTL support and print styles
html_full = f"""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تقرير PCA</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                print-color-adjust: exact;
                -webkit-print-color-adjust: exact;
            }}
        }}
        
        body {{
            font-family: 'Arial', 'DejaVu Sans', 'Tahoma', sans-serif;
            direction: rtl;
            text-align: right;
            line-height: 1.8;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: white;
        }}
        
        h1 {{
            font-size: 2.2em;
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            margin-top: 0;
        }}
        
        h2 {{
            font-size: 1.6em;
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        h3 {{
            font-size: 1.3em;
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 10px;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        ul, ol {{
            padding-right: 30px;
            margin-bottom: 15px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', 'Monaco', monospace;
            font-size: 0.9em;
            color: #c7254e;
        }}
        
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
            overflow-x: auto;
            direction: ltr;
            text-align: left;
            margin: 20px 0;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            color: #333;
        }}
        
        strong {{
            color: #2c3e50;
        }}
        
        @media print {{
            body {{
                max-width: 100%;
                padding: 0;
            }}
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""

# Save HTML
Path('REPORT_PCA_PERSONAL_print.html').write_text(html_full, encoding='utf-8')
print("✅ تم إنشاء ملف HTML جاهز للطباعة!")
print("📄 الملف: REPORT_PCA_PERSONAL_print.html")
print("\n📋 التعليمات:")
print("1. افتح الملف في متصفح (Chrome/Safari)")
print("2. اضغط Cmd+P (أو Ctrl+P)")
print("3. اختر 'Save as PDF'")
print("4. احفظ الملف كـ REPORT_PCA_PERSONAL.pdf")


