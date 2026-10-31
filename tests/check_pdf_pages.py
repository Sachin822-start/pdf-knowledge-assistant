
from pathlib import Path
import fitz

PDF_FOLDER = Path("data/pdfs")

def main():
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    total_pages = 0

    print("\nPDF DATASET VERIFICATION")
    print("=" * 65)

    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as pdf:
                page_count = len(pdf)
                total_pages += page_count

                status = "PASS" if page_count >= 200 else "NEEDS EXPANSION"

                print(
                    f"{pdf_path.name:<48} "
                    f"{page_count:>5} pages  {status}"
                )

        except Exception as error:
            print(f"ERROR: {pdf_path.name}: {error}")

    print("=" * 65)
    print("Total PDF files:", len(pdf_files))
    print("Total pages:", total_pages)
    print("PDFs with 200+ pages:", sum(
        1 for p in pdf_files
        if fitz.open(p).page_count >= 200
    ))

if __name__ == "__main__":
    main()
