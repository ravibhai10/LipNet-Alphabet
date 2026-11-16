import pypdf
import os

pdf_path = r"C:\Users\Dell\Downloads\ISE_syllabus_ESD_July_Nov_2026.pdf"
if os.path.exists(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    print("Num pages:", len(reader.pages))
    text = reader.pages[0].extract_text()
    print("Page 1 preview:\n", text[:1000])
else:
    print("PDF not found")
