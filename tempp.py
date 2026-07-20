import os
import hashlib

# ====== CHANGE THIS TO YOUR FOLDER ======
folder_path = "/home/ammar/Downloads/rvs_moto"
# ========================================

# Supported image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def file_hash(filepath):
    """Return SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

seen_hashes = {}
deleted = 0
kept = 0

for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)

    if not os.path.isfile(filepath):
        continue

    ext = os.path.splitext(filename)[1].lower()
    if ext not in image_extensions:
        continue

    try:
        h = file_hash(filepath)

        if h in seen_hashes:
            print(f"Deleting duplicate: {filename}")
            os.remove(filepath)
            deleted += 1
        else:
            seen_hashes[h] = filename
            kept += 1

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\nDone!")
print(f"Images kept: {kept}")
print(f"Duplicates deleted: {deleted}")