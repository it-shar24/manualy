import sys
from pathlib import Path
from src.ingestion.pdf_parser import PDFManualParser

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_parser.py path/to/manual.pdf")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    
    print(f"\n[Manualy Engine] Ingesting: {pdf_file_path}")
    parser = PDFManualParser(pdf_file_path)
    result = parser.extract()

    print("\n" + "="*50)
    print(f"DOCUMENT ID       : {result['doc_id']}")
    print(f"TOTAL PAGES       : {result['total_pages']}")
    print(f"TOTAL IMAGES FOUND: {result['total_images_extracted']}")
    print("="*50)

    for page in result["pages"][:5]:  # Preview first 5 pages
        print(f"\n--- Page {page['page_number']} ---")
        print(f"Text Snippet : {page['text'][:120].replace(chr(10), ' ')}...")
        print(f"Visual Assets: {page['image_paths']}")

    print("\nExtraction test completed successfully.")