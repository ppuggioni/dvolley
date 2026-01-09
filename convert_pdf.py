import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md, output_pdf):
    # 1. Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Convert to HTML with tables extension
    html_content = markdown.markdown(text, extensions=['tables'])

    # 3. Add CSS for styling
    full_html = f"""
    <html>
    <head>
    <style>
        body {{
            font-family: Helvetica, Arial, sans-serif;
            font-size: 12px;
            line-height: 1.5;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 20px; }}
        h3 {{ color: #7f8c8d; }}
        
        /* Table Styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
            color: #333;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        /* Image Styling */
        img {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
        }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    # 4. Generate PDF
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            full_html,                # the HTML to convert
            dest=result_file          # file handle to recieve result
        )

    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
    else:
        print(f"Successfully created PDF: {output_pdf}")

if __name__ == "__main__":
    source = r"c:\Users\ppaol\.gemini\antigravity\brain\fc253d6d-f6c7-411f-87a1-5ada9b8ad1d7\loocv_calibration_report.md"
    output = r"c:\Users\ppaol\.gemini\antigravity\brain\fc253d6d-f6c7-411f-87a1-5ada9b8ad1d7\loocv_calibration_report_styled.pdf"
    convert_md_to_pdf(source, output)
