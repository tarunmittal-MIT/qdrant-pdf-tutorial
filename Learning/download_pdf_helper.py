"""
Helper script to download Azure AZ-104 PDF
This script helps locate and download the Azure AZ-104 learning materials.
"""

import os
import requests
from pathlib import Path

def find_az104_pdf_urls():
    """
    Provides information about where to find the AZ-104 PDF.
    """
    print("="*70)
    print("Azure AZ-104 Learning PDF Download Guide")
    print("="*70)
    print("\nOfficial Microsoft Resources:")
    print("1. Microsoft Learn AZ-104 Page:")
    print("   https://learn.microsoft.com/en-us/credentials/certifications/exams/az-104/")
    print("\n2. Azure Administrator Learning Path:")
    print("   https://learn.microsoft.com/en-us/training/paths/azure-administrator/")
    print("\n3. Exam Study Guide (if available):")
    print("   Look for 'Study Guide' or 'Exam Preparation' links")
    print("\n4. Microsoft Certification Dashboard:")
    print("   https://learn.microsoft.com/en-us/credentials/")
    print("\n" + "="*70)
    print("\nNote: PDFs may require:")
    print("  - Microsoft account login")
    print("  - Access through Microsoft Learn")
    print("  - May be in multiple formats (HTML, PDF, etc.)")
    print("\n" + "="*70)

def download_pdf_from_url(url, output_path="az104-learning-guide.pdf"):
    """
    Attempt to download PDF from URL.
    
    Note: This may require authentication for Microsoft sites.
    """
    try:
        print(f"\nAttempting to download from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' in content_type.lower() or url.endswith('.pdf'):
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(output_path)
            print(f"✓ Downloaded: {output_path} ({file_size:,} bytes)")
            return output_path
        else:
            print(f"⚠ Response is not a PDF (content-type: {content_type})")
            print("   This might be a redirect page or require authentication.")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"⚠ Download failed: {e}")
        print("\nManual Download Instructions:")
        print("  1. Visit the Microsoft Learn pages listed above")
        print("  2. Look for downloadable PDF or study materials")
        print(f"  3. Save the file as: {output_path}")
        return None

if __name__ == "__main__":
    find_az104_pdf_urls()
    
    # Try some common URL patterns (may not work without authentication)
    print("\n" + "="*70)
    print("Attempting Automatic Download")
    print("="*70)
    
    # Note: These URLs are examples and may not work directly
    # Users will likely need to download manually
    test_urls = [
        # Add any direct PDF URLs here if known
    ]
    
    if test_urls:
        for url in test_urls:
            result = download_pdf_from_url(url)
            if result:
                break
    else:
        print("\n⚠ No direct download URLs configured.")
        print("   Please download the PDF manually from Microsoft Learn.")
        print("   Once downloaded, save it as 'az104-learning-guide.pdf' in the current directory.")

