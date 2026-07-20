import os
from PIL import Image
import imagehash

folder = os.path.expanduser("~/Downloads/rvs_moto")

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

seen = {}
deleted = 0

for filename in os.listdir(folder):
    if not filename.lower().endswith(extensions):
        continue

    path = os.path.join(folder, filename)

    try:
        img = Image.open(path)
        h = imagehash.phash(img)   # perceptual hash

        if h in seen:
            print(f"Deleting duplicate: {filename}")
            os.remove(path)
            deleted += 1
        else:
            seen[h] = filename

    except Exception as e:
        print(f"Skipping {filename}: {e}")

print(f"\nDeleted {deleted} duplicates.")
print(f"Remaining unique images: {len(seen)}")