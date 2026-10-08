from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FILE = "NECA_Website_Knowledge.txt"


def extract_pages(file_path):
    """Read PDF, DOCX or TXT and return (page_number, text) pairs."""
    path = Path(file_path).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path.name}")

    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Please use a document smaller than 20 MB.")

    extension = path.suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(str(path))

        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("Please use a PDF without a password.")

        pages = []

        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                pages.append((number, text))
            else:
                print(f"Notice: PDF page {number} has no readable text.")

    elif extension == ".docx":
        document = Document(str(path))
        parts = []

        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                if block.text.strip():
                    parts.append(block.text.strip())
            else:
                for row in block.rows:
                    parts.append(
                        " | ".join(cell.text.strip() for cell in row.cells)
                    )

        text = "\n".join(parts).strip()
        pages = [(0, text)] if text else []

    elif extension == ".txt":
        text = path.read_text(encoding="utf-8-sig").strip()
        pages = [(0, text)] if text else []

    else:
        raise ValueError("Supported formats are .pdf, .docx and .txt.")

    if not pages:
        raise ValueError(
            "No readable text found. An image-only PDF needs OCR."
        )

    return pages


def save_extracted_text(file_path, pages):
    output_folder = BASE_DIR / "extracted"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / (
        f"{Path(file_path).stem}_extracted.txt"
    )

    sections = []

    for number, text in pages:
        heading = f"[Page {number}]" if number else "[Document text]"
        sections.append(f"{heading}\n{text}")

    output_path.write_text(
        "\n\n".join(sections),
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    try:
        print("NECA Document Processor")
        print(f"Press Enter to process {DEFAULT_FILE}.")
        print("Or paste a path to another PDF, DOCX or TXT document.\n")

        file_path = input("Document: ").strip().strip('"').strip("'")
        file_path = file_path or DEFAULT_FILE

        pages = extract_pages(file_path)
        output_path = save_extracted_text(file_path, pages)
        word_count = sum(len(text.split()) for _, text in pages)

        print("\nSUCCESS: Document text extracted")
        print(f"Words extracted: {word_count}")
        print(f"Text file saved to: {output_path}")

    except Exception as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)