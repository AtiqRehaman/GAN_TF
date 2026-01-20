from PIL import Image
from tqdm import tqdm
import os
import argparse
import random


def folder_resize(src_dir, outdir, dimensions, n=None, random_select=False):
    """Resize up to `n` images from `src_dir` and save into `outdir`.

    - `dimensions` should be a (width, height) tuple of ints.
    - If `n` is None or greater than available files, all images are processed.
    - If `random_select` is True, a random sample of `n` files is chosen.
    """
    os.makedirs(outdir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    total = len(files)

    if total == 0:
        print("No image files found in the source folder.")
        return

    if n is None or n >= total:
        chosen = files
    else:
        if random_select:
            chosen = random.sample(files, n)
        else:
            chosen = files[:n]

    print(f"Resizing {len(chosen)} images from '{src_dir}' -> '{outdir}' to {dimensions}")

    for file in tqdm(chosen, desc="Processing images", unit="img"):
        src_path = os.path.join(src_dir, file)
        name_no_ext = os.path.splitext(file)[0]
        dst_path = os.path.join(outdir, name_no_ext + '.png')
        try:
            with Image.open(src_path) as img:
                resized_img = img.resize(dimensions, resample=Image.LANCZOS)
                # Save as PNG
                resized_img.save(dst_path, format='PNG')
        except Exception as e:
            tqdm.write(f"Skipped {file}: {e}")
            continue

    print("\nImage resizing completed successfully.")


def file_resize(file_path, dimensions, outpath=None):
    """Resize a single image file and overwrite or save to `outpath`."""
    try:
        if outpath is None:
            outpath = file_path

        with Image.open(file_path) as img:
            resized = img.resize(dimensions, resample=Image.LANCZOS)
            # Ensure output path has .png extension
            if outpath is None:
                outpath = os.path.splitext(file_path)[0] + '.png'
            else:
                base, ext = os.path.splitext(outpath)
                if ext.lower() != '.png':
                    outpath = base + '.png'
            resized.save(outpath, format='PNG')

        print("Image resize completed.")

    except Exception as e:
        print(f"Error resizing file: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Resize images in a folder or a single file")
    sub = p.add_subparsers(dest='mode', required=True)

    # Folder mode
    f = sub.add_parser('folder', help='Resize images in a folder')
    f.add_argument('--src', required=True, help='Source folder path')
    f.add_argument('--dst', required=True, help='Destination folder path')
    f.add_argument('--width', type=int, required=True, help='Target width')
    f.add_argument('--height', type=int, required=True, help='Target height')
    f.add_argument('--n', type=int, default=None, help='Number of images to process (default: all)')
    f.add_argument('--random', action='store_true', help='Randomly select n images')

    # File mode
    g = sub.add_parser('file', help='Resize a single file')
    g.add_argument('file', help='Image file path')
    g.add_argument('--width', type=int, required=True, help='Target width')
    g.add_argument('--height', type=int, required=True, help='Target height')
    g.add_argument('--out', help='Output file path (if omitted will overwrite)')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'folder':
        dims = (args.width, args.height)
        folder_resize(args.src, args.dst, dims, n=args.n, random_select=args.random)
    elif args.mode == 'file':
        dims = (args.width, args.height)
        file_resize(args.file, dims, outpath=args.out)
