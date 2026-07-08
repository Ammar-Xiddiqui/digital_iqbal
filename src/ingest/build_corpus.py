import os
import yaml
import json
import unicodedata
import hashlib
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw/iqbal-demystified-dataset/data/poems")
PROCESSED_FILE = Path("data/processed/corpus.jsonl")
REPORT_FILE = Path("data_quality_report.md")

# Map directory prefixes to standard book names
BOOK_MAP = {
    "001": ("Bang-e-Dra", "ur"),
    "002": ("Bal-e-Jibril", "ur"),
    "003": ("Zarb-e-Kalim", "ur"),
    "004": ("Armughan-e-Hijaz (Urdu)", "ur"),
    "005": ("Asrar-e-Khudi", "fa"),
    "006": ("Rumuz-e-Bekhudi", "fa"),
    "007": ("Payam-e-Mashriq", "fa"),
    "008": ("Zabur-e-Ajam", "fa"),
    "009": ("Javid Nama", "fa"),
    "010": ("Pas Cheh Bayad Kard", "fa"),
    "011": ("Armughan-e-Hijaz (Persian)", "fa")
}

def normalize_text(text):
    """Normalizes Unicode to NFC and cleans basic whitespace."""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFC', text)
    return " ".join(normalized.split())

def hash_text(text):
    """Generates a hash for deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def build_corpus():
    """Parses YAML files, builds JSONL, and tracks metrics."""
    corpus = []
    seen_hashes = set()
    seen_ids = set()
    
    stats = {
        "total_verses": 0,
        "duplicates_removed": 0,
        "books": {},
        "languages": {"ur": 0, "fa": 0, "unknown": 0}
    }

    # Iterate through all YAML files in the poems directory
    for filepath in RAW_DIR.rglob("*.yaml"):
        book_id = filepath.parent.name
        book_info = BOOK_MAP.get(book_id, ("Unknown Book", "unknown"))
        book_name, default_lang = book_info

        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                if not data or 'sher' not in data:
                    continue
                
                # Extract Poem Title (Prefer Urdu, fallback to English)
                poem_title = "Unknown Poem"
                for heading in data.get('heading', []):
                    if heading.get('lang') == 'ur':
                        poem_title = heading.get('text')
                        break
                    elif heading.get('lang') == 'en':
                        poem_title = heading.get('text')

                verse_number = 1
                for item in data.get("sher", []):
                    verse_id = item.get("id", f"UNKNOWN-{stats['total_verses']}")
                    
                    text_original = ""
                    text_roman = ""
                    translation_en = None
                    
                    # Extract languages from sherContent
                    for content in item.get("sherContent", []):
                        cl = content.get("lang")
                        ct = content.get("text", "")
                        
                        if cl == "ur": 
                            # Note: The dataset often tags Persian text as 'ur', so we 
                            # capture it here but use the book mapping to define the true language.
                            text_original = normalize_text(ct)
                        elif cl == "ro":
                            text_roman = normalize_text(ct)
                        elif cl == "en":
                            translation_en = normalize_text(ct)
                            
                    if not text_original:
                        continue # Skip empty verses
                        
                    text_hash = hash_text(text_original)
                    
                    # Deduplication checks
                    if text_hash in seen_hashes or verse_id in seen_ids:
                        stats["duplicates_removed"] += 1
                        continue
                        
                    seen_hashes.add(text_hash)
                    seen_ids.add(verse_id)
                    
                    verse_record = {
                        "verse_id": verse_id,
                        "book": book_name,
                        "poem_title": poem_title,
                        "verse_number_in_poem": verse_number,
                        "language": default_lang,
                        "text_original": text_original,
                        "text_roman": text_roman,
                        "translation_en": translation_en,
                        "theme_tags": [], # Tags don't seem present in this YAML level
                        "source_url": f"https://github.com/AzeemGhumman/iqbal-demystified-dataset/tree/master/data/poems/{book_id}/{filepath.name}"
                    }
                    
                    corpus.append(verse_record)
                    stats["total_verses"] += 1
                    stats["books"][book_name] = stats["books"].get(book_name, 0) + 1
                    stats["languages"][default_lang] = stats["languages"].get(default_lang, 0) + 1
                    
                    verse_number += 1

            except yaml.YAMLError:
                print(f"Skipping invalid YAML file: {filepath}")

    # Write JSONL
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_FILE, 'w', encoding='utf-8') as f:
        for record in corpus:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    return stats

def generate_report(stats):
    """Generates the data quality markdown report."""
    report_content = f"""# Data Quality Report

## Summary
* **Total Clean Verses:** {stats['total_verses']}
* **Duplicates Removed:** {stats['duplicates_removed']}

## Language Split
"""
    for lang, count in stats['languages'].items():
        report_content += f"* **{lang.upper()}:** {count}\n"
        
    report_content += "\n## Book Coverage\n"
    for book, count in stats['books'].items():
        report_content += f"* **{book}:** {count} verses\n"
        
    report_content += """
## Known Gaps & Next Steps
* Verify Persian works (Asrar-e-Khudi, Payam-e-Mashriq, Zabur-e-Ajam) against Iqbal Cyber Library.
* Conduct 50-verse manual spot check.
"""
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Report generated at {REPORT_FILE}")

if __name__ == "__main__":
    if not RAW_DIR.exists():
        print(f"Error: Raw directory {RAW_DIR} not found. Ensure dataset is cloned.")
    else:
        print("Parsing YAML and building corpus...")
        stats = build_corpus()
        generate_report(stats)
        print("Phase 1 extraction complete. Please check data_quality_report.md.")