"""
Quick script to extract text from Red Hat RH294 PDF for viewing in Cursor
"""
import os
import sys

def extract_pdf_to_text(pdf_path, output_path=None):
    """Extract text from PDF and save to a text file."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("⚠ PyMuPDF not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "pymupdf"], check=True)
        import fitz
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return False
    
    if output_path is None:
        # Create output filename based on PDF name
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"{base_name}_extracted.txt"
    
    print(f"📖 Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    print(f"📄 Total pages: {len(doc)}")
    print("🔍 Extracting text...")
    
    full_text = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        full_text.append(f"\n{'='*70}\nPAGE {page_num}\n{'='*70}\n\n{text}\n")
        
        if page_num % 50 == 0:
            print(f"  Processed {page_num}/{len(doc)} pages...")
    
    doc.close()
    
    # Write to file
    print(f"💾 Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_text))
    
    file_size = os.path.getsize(output_path)
    total_pages = len(full_text)
    print(f"✅ Extraction complete!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size / (1024*1024):.2f} MB")
    print(f"   Pages: {total_pages}")
    
    return True

if __name__ == "__main__":
    # Try to find the PDF
    pdf_paths = [
        "../rh294-9.0-student-guide (6).pdf",
        "rh294-9.0-student-guide (6).pdf",
        os.path.expanduser("~/rh294-9.0-student-guide (6).pdf"),
    ]
    
    pdf_path = None
    for path in pdf_paths:
        if os.path.exists(path):
            pdf_path = path
            break
    
    if not pdf_path:
        print("❌ PDF not found. Please specify the path:")
        pdf_path = input("Enter PDF path: ").strip()
    
    if pdf_path and os.path.exists(pdf_path):
        extract_pdf_to_text(pdf_path)
    else:
        print("❌ Invalid PDF path")

