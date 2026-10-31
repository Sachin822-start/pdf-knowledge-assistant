
import pymupdf
import pytesseract
import shutil
from PIL import Image
from pathlib import Path


# --------------------------------------------------
# TESSERACT CONFIGURATION
# --------------------------------------------------

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
else:
    raise RuntimeError(
        "Tesseract OCR not found. Please install Tesseract."
    )


# --------------------------------------------------
# OCR FUNCTION
# --------------------------------------------------

def perform_ocr(page, dpi=250):

    # Convert PDF page into an image
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    # Extract text using Tesseract
    text = pytesseract.image_to_string(
        image,
        lang="eng"
    )

    return text.strip()


# --------------------------------------------------
# PDF EXTRACTION FUNCTION
# --------------------------------------------------

def extract_pdf(pdf_path, min_native_chars=30):

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = []

    pdf = pymupdf.open(pdf_path)

    print(f"\nProcessing: {pdf_path.name}")
    print(f"Total pages: {len(pdf)}")

    for page_number, page in enumerate(pdf, start=1):

        # Attempt native text extraction
        text = page.get_text("text").strip()

        extraction_method = "native"

        # Apply OCR if native text is insufficient
        if len(text) < min_native_chars:

            print(f"OCR processing page {page_number}")

            text = perform_ocr(page)

            extraction_method = "ocr"

        # Preserve page-level metadata
        if text:

            documents.append({
                "text": text,
                "metadata": {
                    "pdf_id": pdf_path.stem,
                    "filename": pdf_path.name,
                    "page_number": page_number,
                    "extraction_method": extraction_method
                }
            })

    pdf.close()

    print(f"Successfully extracted pages: {len(documents)}")

    return documents


# --------------------------------------------------
# TEST THE PDF LOADER
# --------------------------------------------------

if __name__ == "__main__":

    pdf_folder = Path("data/pdfs")

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/pdfs")
        print("Please add a sample PDF and try again.")

    else:

        result = extract_pdf(pdf_files[0])

        print("\n========== EXTRACTION RESULT ==========")

        for item in result[:2]:

            print("\nMetadata:")
            print(item["metadata"])

            print("\nExtracted text:")
            print(item["text"][:500])

        print("\nPDF extraction test completed!")
