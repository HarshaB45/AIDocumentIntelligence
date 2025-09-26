from pathlib import Path
from pdfminer.high_level import extract_text

PDF_ROOT = Path("full_contract_pdf")
PARTS = ["Part_I", "Part_II"]   # process all three
TXT_OUT = Path("full_contract2_txt")
TXT_OUT.mkdir(parents=True, exist_ok=True)

def flat_name(rel_path: Path) -> str:
    """
    Convert nested path like 'Part_I/Transportation/file.pdf' into:
    'Part_I__Transportation__file.txt'
    """
    *folders, filename = rel_path.parts
    return "__".join(folders + [Path(filename).stem]) + ".txt"

def main():
    all_pdfs = []
    for part in PARTS:
        part_dir = PDF_ROOT / part
        if not part_dir.exists():
            print(f"⚠️ Skipping missing folder: {part_dir}")
            continue
        all_pdfs.extend(sorted(part_dir.rglob("*.pdf")))

    if not all_pdfs:
        raise SystemExit(f"No PDFs found under {PDF_ROOT}")

    print(f"Found {len(all_pdfs)} PDFs under {PDF_ROOT}")
    for pdf in all_pdfs:
        try:
            rel = pdf.relative_to(PDF_ROOT)   # e.g. Part_I/Transportation/file.pdf
            out_name = flat_name(rel)
            out_path = TXT_OUT / out_name

            if out_path.exists():
                print(f"↷ skip (exists): {out_path.name}")
                continue

            text = extract_text(str(pdf)) or ""
            if len(text.strip()) < 50:
                print(f"⚠️ very short text after conversion: {pdf.name}")

            out_path.write_text(text, encoding="utf-8")
            print(f"✓ {pdf} -> {out_path.name}")
        except Exception as e:
            print(f"✗ Failed: {pdf} ({e})")

if __name__ == "__main__":
    main()
