from pathlib import Path
import json
import fitz  # PyMuPDF


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
TEXT_DIR = BASE_DIR / "data" / "processed" / "extracted_text"
TABLE_DIR = BASE_DIR / "data" / "processed" / "tables"


def extract_pdf(pdf_path: Path):
    """
    PDF içerisindeki:
    - sayfa metinlerini
    - tabloları
    - görselleri

    tespit eder ve yapılandırılmış veri döndürür.
    """

    document = fitz.open(pdf_path)

    pages = []

    total_tables = 0
    total_images = 0

    for page_number, page in enumerate(document, start=1):

        # -------------------------
        # TEXT
        # -------------------------

        text = page.get_text("text").strip()

        # -------------------------
        # TABLES
        # -------------------------

        tables = []

        try:
            table_finder = page.find_tables()

            for table_index, table in enumerate(
                table_finder.tables,
                start=1
            ):

                rows = table.extract()

                table_data = {
                    "table_index": table_index,
                    "rows": rows
                }

                tables.append(table_data)

                total_tables += 1

        except Exception as error:
            print(
                f"[WARNING] Table extraction failed "
                f"{pdf_path.name} - page {page_number}: {error}"
            )

        # -------------------------
        # IMAGES
        # -------------------------

        images = page.get_images(full=True)

        image_data = []

        for image_index, image in enumerate(images, start=1):

            image_data.append({
                "image_index": image_index,
                "xref": image[0]
            })

        total_images += len(image_data)

        # -------------------------
        # PAGE DATA
        # -------------------------

        pages.append({
            "page": page_number,
            "text": text,
            "tables": tables,
            "images": image_data
        })

    document.close()

    return {
        "document": pdf_path.name,
        "page_count": len(pages),
        "table_count": total_tables,
        "image_count": total_images,
        "pages": pages
    }


def process_all_pdfs():

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("[ERROR] data/raw klasöründe PDF bulunamadı.")
        return

    print("=" * 60)
    print(f"Bulunan PDF sayısı: {len(pdf_files)}")
    print("=" * 60)

    for pdf_path in pdf_files:

        print()
        print(f"İşleniyor: {pdf_path.name}")

        result = extract_pdf(pdf_path)

        output_file = TEXT_DIR / f"{pdf_path.stem}.json"

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(f"  Sayfa sayısı : {result['page_count']}")
        print(f"  Tablo sayısı : {result['table_count']}")
        print(f"  Görsel sayısı: {result['image_count']}")
        print(f"  Çıktı        : {output_file}")

    print()
    print("=" * 60)
    print("PDF INGESTION TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    process_all_pdfs()