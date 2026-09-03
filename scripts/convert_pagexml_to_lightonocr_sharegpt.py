#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAGE-XML / ALTO-XML to ShareGPT Format Converter for LightOnOCR-2

Converts PAGE-XML or ALTO-XML OCR datasets to the ShareGPT format expected by
LlamaFactory for training LightOnOCR-2 (https://huggingface.co/lightonai/LightOnOCR-2-1B)
with LoRA or full fine-tuning.

LightOnOCR-2 differs from GLM-OCR in that:
  - The user prompt contains ONLY the image placeholder (no "Text Recognition:" text).
    The OCR behavior is embedded in the model weights.
  - The model natively outputs Markdown with LaTeX math spans.
  - Vision encoder is Pixtral-based (not GLM4V-based).

Output format (ShareGPT for multimodal SFT):
{
  "messages": [
    {"role": "user", "content": "<image>"},
    {"role": "assistant", "content": "transcribed text"}
  ],
  "images": ["dataset_name/unique_id.png"]
}

Image paths are relative to LlamaFactory's data/ directory. Place the output
JSON and image folder in data/ and register in dataset_info.json.

Usage:
    python convert_pagexml_to_lightonocr_sharegpt.py \\
        --input_dir /mnt/d/datasets/cmmhwr26 \\
        --output_dir ./data \\
        --dataset_name cmmhwr26_lightonocr \\
        --format auto \\
        --unicode_form NFC
"""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# PAGE-XML namespace schemas
PAGE_NAMESPACES = {
    "2009-03-16": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2009-03-16",
    "2010-01-12": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-01-12",
    "2010-03-19": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19",
    "2013-07-15": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
    "2014-08-26": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2014-08-26",
    "2016-07-15": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2016-07-15",
    "2017-07-15": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15",
    "2018-07-15": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15",
    "2019-07-15": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
}

# ALTO-XML namespace schemas
ALTO_NAMESPACES = {
    "v2": "http://www.loc.gov/standards/alto/ns-v2#",
    "v3": "http://www.loc.gov/standards/alto/ns-v3#",
    "v4": "http://www.loc.gov/standards/alto/ns-v4#",
}


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode text."""
    import unicodedata
    if not text:
        return ""
    return unicodedata.normalize(form, text.strip())


def detect_xml_format(xml_file: Path) -> Tuple[str, str]:
    """
    Detect XML format (PAGE-XML or ALTO-XML) and namespace.
    Returns (format_type, namespace).
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        namespace = None
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0][1:]
        else:
            xmlns = root.get("{http://www.w3.org/2000/xmlns/}xmlns")
            if xmlns:
                namespace = xmlns

        if namespace:
            for _, alto_ns in ALTO_NAMESPACES.items():
                if namespace == alto_ns:
                    return ("alto", alto_ns)
            for _, alto_ns in ALTO_NAMESPACES.items():
                if alto_ns in namespace or "alto" in namespace.lower():
                    return ("alto", ALTO_NAMESPACES["v4"])

        if namespace:
            for _, page_ns in PAGE_NAMESPACES.items():
                if namespace == page_ns:
                    return ("pagexml", page_ns)
            for _, page_ns in PAGE_NAMESPACES.items():
                if page_ns in namespace or "page" in namespace.lower():
                    return ("pagexml", PAGE_NAMESPACES["2019-07-15"])

        root_tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if root_tag.lower() == "alto":
            return ("alto", ALTO_NAMESPACES["v4"])
        if "page" in root_tag.lower():
            return ("pagexml", PAGE_NAMESPACES["2019-07-15"])

        return ("pagexml", PAGE_NAMESPACES["2019-07-15"])
    except Exception as e:
        print(f"Warning: Could not detect format for {xml_file}: {e}")
        return ("pagexml", PAGE_NAMESPACES["2019-07-15"])


def parse_polygon_coords(polygon_str: str, format_type: str = "pagexml") -> List[Tuple[int, int]]:
    """Parse polygon coordinate string to list of (x, y) tuples."""
    if not polygon_str:
        return []

    points = []
    if format_type == "alto":
        coords = polygon_str.strip().split()
        for i in range(0, len(coords) - 1, 2):
            try:
                points.append((int(float(coords[i])), int(float(coords[i + 1]))))
            except (ValueError, IndexError):
                continue
    else:
        for coord in polygon_str.strip().split():
            try:
                if "," in coord:
                    x, y = coord.split(",", 1)
                    points.append((int(float(x)), int(float(y))))
            except (ValueError, IndexError):
                continue
    return points


def extract_textlines_from_pagexml(
    xml_file: Path, namespace: str
) -> List[Dict[str, Any]]:
    """Extract textline polygon coordinates and transcriptions from PAGE-XML."""
    textlines = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for textline in root.findall(f".//{{{namespace}}}TextLine"):
            coords_elem = textline.find(f"{{{namespace}}}Coords")
            if coords_elem is None:
                continue
            polygon_str = coords_elem.get("points")
            if not polygon_str:
                continue
            coords = parse_polygon_coords(polygon_str, "pagexml")
            if len(coords) < 3:
                continue
            text_equiv = textline.find(f"{{{namespace}}}TextEquiv")
            transcription = ""
            if text_equiv is not None:
                unicode_elem = text_equiv.find(f"{{{namespace}}}Unicode")
                if unicode_elem is not None and unicode_elem.text:
                    transcription = unicode_elem.text.strip()
            if transcription:
                textlines.append({"coords": coords, "transcription": transcription})
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
    return textlines


def extract_textlines_from_alto(xml_file: Path, namespace: str) -> List[Dict[str, Any]]:
    """Extract textline polygon coordinates and transcriptions from ALTO-XML."""
    textlines = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for textline in root.findall(f".//{{{namespace}}}TextLine"):
            coords = []
            shape_elem = textline.find(f"{{{namespace}}}Shape")
            if shape_elem is not None:
                polygon_elem = shape_elem.find(f"{{{namespace}}}Polygon")
                if polygon_elem is not None and polygon_elem.get("POINTS"):
                    coords = parse_polygon_coords(polygon_elem.get("POINTS"), "alto")
            if not coords or len(coords) < 3:
                try:
                    hpos = int(textline.get("HPOS", 0))
                    vpos = int(textline.get("VPOS", 0))
                    width = int(textline.get("WIDTH", 0))
                    height = int(textline.get("HEIGHT", 0))
                    if width > 0 and height > 0:
                        coords = [
                            (hpos, vpos),
                            (hpos + width, vpos),
                            (hpos + width, vpos + height),
                            (hpos, vpos + height),
                        ]
                except (ValueError, TypeError):
                    pass
            transcription_parts = []
            for s in textline.findall(f"{{{namespace}}}String"):
                content = s.get("CONTENT", "")
                if content:
                    transcription_parts.append(content)
            transcription = " ".join(transcription_parts).strip()
            if not transcription and textline.text:
                transcription = textline.text.strip()
            if coords and len(coords) >= 3 and transcription:
                textlines.append({"coords": coords, "transcription": transcription})
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
    return textlines


def get_image_filename_from_pagexml(xml_file: Path, namespace: str) -> Optional[str]:
    """Extract referenced image filename from PAGE-XML Page imageFilename attribute."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        page = root.find(f".//{{{namespace}}}Page")
        if page is not None:
            fn = page.get("imageFilename")
            if fn and fn.strip():
                return fn.strip()
    except Exception:
        pass
    return None


def get_image_filename_from_alto(xml_file: Path, namespace: str) -> Optional[str]:
    """Extract referenced image filename from ALTO sourceImageInformation."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        desc = root.find(f".//{{{namespace}}}Description")
        if desc is None:
            return None
        sii = desc.find(f"{{{namespace}}}sourceImageInformation")
        if sii is None:
            return None
        fn = sii.find(f"{{{namespace}}}fileName")
        if fn is not None and fn.text:
            return fn.text.strip()
    except Exception:
        pass
    return None


def find_image_file(
    xml_file: Path,
    alto_namespace: Optional[str] = None,
    page_namespace: Optional[str] = None,
    image_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Find corresponding image file for an XML file.
    Searches: same dir as XML, then (if image_dir set) image_dir/relative_path.
    Uses imageFilename from PAGE-XML or ALTO when available (handles merged datasets
    where one subdataset uses .jpg and another uses .png).
    """
    search_dirs: List[Path] = [xml_file.parent]
    if image_dir is not None and input_dir is not None:
        try:
            rel = xml_file.parent.relative_to(input_dir)
            search_dirs.insert(0, image_dir / rel)
        except ValueError:
            search_dirs.insert(0, image_dir)

    def try_filename(fn: str) -> Optional[Path]:
        """Try exact filename, then base+ext variants."""
        for base_dir in search_dirs:
            candidate = base_dir / fn
            if candidate.exists():
                return candidate
            base = Path(fn).stem
            for ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".JPG", ".JPEG", ".PNG", ".TIFF", ".TIF"]:
                candidate = base_dir / (base + ext)
                if candidate.exists():
                    return candidate
        return None

    # PAGE-XML: try imageFilename first (handles merged jpg/png datasets)
    if page_namespace:
        fn = get_image_filename_from_pagexml(xml_file, page_namespace)
        if fn:
            result = try_filename(fn)
            if result:
                return result

    # Use stem + ext (not with_suffix) to handle names with dots, e.g. BCUF_Ms._L_2057_027.xml
    stem = xml_file.stem
    for base_dir in search_dirs:
        for ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".JPG", ".JPEG", ".PNG", ".TIFF", ".TIF"]:
            candidate = base_dir / (stem + ext)
            if candidate.exists():
                return candidate

    # ALTO: try sourceImageInformation/fileName
    if alto_namespace:
        fn = get_image_filename_from_alto(xml_file, alto_namespace)
        if fn:
            result = try_filename(fn)
            if result:
                return result
    return None


def crop_image_from_polygon(
    image: Image.Image,
    polygon: List[Tuple[int, int]],
    padding: int = 5,
) -> Image.Image:
    """Crop image region using polygon mask (keeps polygon area, white elsewhere)."""
    if not polygon or len(polygon) < 3:
        return image
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    x_min = max(0, min(xs) - padding)
    y_min = max(0, min(ys) - padding)
    x_max = min(image.width, max(xs) + padding)
    y_max = min(image.height, max(ys) + padding)
    adjusted = [(x - x_min, y - y_min) for x, y in polygon]
    cropped = image.crop((x_min, y_min, x_max, y_max))
    mask = Image.new("L", cropped.size, 0)
    ImageDraw.Draw(mask).polygon(adjusted, fill=255)
    background = Image.new("RGB", cropped.size, (255, 255, 255))
    return Image.composite(cropped, background, mask)


def merge_polygons_to_surrounding(
    polygons: List[List[Tuple[int, int]]]
) -> List[Tuple[int, int]]:
    """Merge polygons into one surrounding polygon (convex hull or bbox)."""
    all_points = []
    for p in polygons:
        if p and len(p) >= 3:
            all_points.extend(p)
    if len(all_points) < 3:
        return []
    if SCIPY_AVAILABLE:
        try:
            arr = np.array(all_points)
            hull = ConvexHull(arr)
            return [tuple(arr[v]) for v in hull.vertices]
        except Exception:
            pass
    xs, ys = [p[0] for p in all_points], [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def group_lines_into_paragraphs(
    textlines: List[Dict[str, Any]],
    min_lines: int = 5,
    max_lines: int = 10,
) -> List[List[Dict[str, Any]]]:
    """Group consecutive lines into paragraphs."""
    if len(textlines) < min_lines:
        return []
    paragraphs = []
    i = 0
    while i < len(textlines):
        sz = random.randint(min_lines, max_lines)
        group = textlines[i : i + sz]
        if len(group) >= min_lines:
            paragraphs.append(group)
        i += sz
    return paragraphs


def unique_image_id(xml_stem: str, sample_type: str, index: int) -> str:
    """Generate a unique image filename to avoid collisions."""
    raw = f"{xml_stem}_{sample_type}_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12] + ".png"


def convert_to_sharegpt(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    format_type: str = "auto",
    unicode_form: str = "NFC",
    min_text_length: int = 1,
    min_crop_size: int = 32,
    include_full_pages: bool = True,
    include_paragraphs: bool = True,
    paragraph_min_lines: int = 5,
    paragraph_max_lines: int = 10,
    line_separator: str = "\n",
    batch_size: int = 100,
    image_format: str = "png",
    symlink_images: bool = False,
    image_dir: Optional[str] = None,
    verbose: bool = False,
    max_files: Optional[int] = None,
) -> None:
    """
    Convert PAGE-XML or ALTO-XML dataset to ShareGPT format for LightOnOCR-2.

    Unlike GLM-OCR, the user message contains only the image placeholder
    (no "Text Recognition:" prompt) because LightOnOCR-2 is trained to
    perform OCR from images without explicit task instructions.

    Memory-efficient: processes files in batches, writes images immediately,
    keeps only small sample dicts in memory.
    """
    image_dir_path = Path(image_dir) if image_dir else None

    if not PIL_AVAILABLE:
        print("Error: PIL/Pillow is required. pip install Pillow")
        sys.exit(1)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    images_dir = output_path / dataset_name
    images_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(
        list(input_path.rglob("*.xml")) + list(input_path.rglob("*.XML"))
    )
    if max_files is not None:
        xml_files = xml_files[:max_files]
        print(f"Limiting to first {len(xml_files)} files (--max-files)")
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return

    # LightOnOCR-2: user content is only the image placeholder, no text prompt
    user_content = "<image>"

    samples: List[Dict[str, Any]] = []
    skipped = 0
    line_count = paragraph_count = page_count = 0

    def progress(iterable, desc):
        return tqdm(iterable, desc=desc) if TQDM_AVAILABLE else iterable

    for batch_start in progress(range(0, len(xml_files), batch_size), "Converting"):
        batch_end = min(batch_start + batch_size, len(xml_files))
        batch = xml_files[batch_start:batch_end]

        for xml_file in batch:
            try:
                fmt, namespace = detect_xml_format(xml_file)
                alto_ns = namespace if fmt == "alto" else None
                page_ns = namespace if fmt == "pagexml" else None
                image_file = find_image_file(
                    xml_file,
                    alto_namespace=alto_ns,
                    page_namespace=page_ns,
                    image_dir=image_dir_path,
                    input_dir=input_path,
                )
                if not image_file:
                    skipped += 1
                    if verbose:
                        print(f"[SKIP] No image found for: {xml_file}")
                    continue

                try:
                    full_image = Image.open(image_file).convert("RGB")
                except Exception as e:
                    if verbose:
                        print(f"[SKIP] Could not load {image_file}: {e}")
                    skipped += 1
                    continue

                if fmt == "alto":
                    textlines = extract_textlines_from_alto(xml_file, namespace)
                else:
                    textlines = extract_textlines_from_pagexml(xml_file, namespace)

                if not textlines:
                    skipped += 1
                    if verbose:
                        print(f"[SKIP] No textlines in: {xml_file}")
                    del full_image
                    gc.collect()
                    continue

                page_transcriptions: List[str] = []
                sample_idx = 0

                # Line-level samples
                for line_data in textlines:
                    trans = normalize_unicode(line_data["transcription"], unicode_form)
                    if len(trans) < min_text_length:
                        continue
                    page_transcriptions.append(trans)
                    try:
                        cropped = crop_image_from_polygon(
                            full_image, line_data["coords"], padding=5
                        )
                    except Exception:
                        continue
                    if cropped.width < min_crop_size or cropped.height < min_crop_size:
                        continue
                    img_id = unique_image_id(xml_file.stem, "line", sample_idx)
                    img_path = images_dir / img_id
                    cropped.save(img_path, format=image_format.upper())
                    rel_path = f"{dataset_name}/{img_id}"
                    samples.append({
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": trans},
                        ],
                        "images": [rel_path],
                    })
                    sample_idx += 1
                    line_count += 1
                    del cropped

                # Paragraph-level samples
                if include_paragraphs and len(textlines) >= paragraph_min_lines:
                    for pg_group in group_lines_into_paragraphs(
                        textlines, paragraph_min_lines, paragraph_max_lines
                    ):
                        trans_parts = []
                        polys = []
                        for ld in pg_group:
                            t = normalize_unicode(ld["transcription"], unicode_form)
                            if len(t) >= min_text_length:
                                trans_parts.append(t)
                                polys.append(ld["coords"])
                        if len(trans_parts) < paragraph_min_lines:
                            continue
                        surrounding = merge_polygons_to_surrounding(polys)
                        if len(surrounding) < 3:
                            continue
                        try:
                            pg_img = crop_image_from_polygon(
                                full_image, surrounding, padding=5
                            )
                        except Exception:
                            continue
                        if pg_img.width < min_crop_size or pg_img.height < min_crop_size:
                            continue
                        pg_text = line_separator.join(trans_parts)
                        if len(pg_text) < min_text_length:
                            continue
                        img_id = unique_image_id(xml_file.stem, "para", sample_idx)
                        img_path = images_dir / img_id
                        pg_img.save(img_path, format=image_format.upper())
                        rel_path = f"{dataset_name}/{img_id}"
                        samples.append({
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": pg_text},
                            ],
                            "images": [rel_path],
                        })
                        sample_idx += 1
                        paragraph_count += 1
                        del pg_img

                # Full-page samples
                if include_full_pages and page_transcriptions:
                    full_text = line_separator.join(page_transcriptions)
                    full_text = normalize_unicode(full_text, unicode_form)
                    if len(full_text) >= min_text_length:
                        img_id = unique_image_id(xml_file.stem, "page", 0)
                        img_path = images_dir / img_id
                        if symlink_images and image_file.exists():
                            if img_path.exists():
                                img_path.unlink()
                            img_path.symlink_to(os.path.abspath(image_file))
                        else:
                            full_image.save(img_path, format=image_format.upper())
                        rel_path = f"{dataset_name}/{img_id}"
                        samples.append({
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": full_text},
                            ],
                            "images": [rel_path],
                        })
                        page_count += 1

                del full_image
                gc.collect()

            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                skipped += 1
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue

    if not samples:
        print("No samples created. Check input directory and image paths.")
        return

    random.shuffle(samples)
    json_path = output_path / f"{dataset_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\nConversion complete.")
    print(f"  Samples: {len(samples):,} (line: {line_count}, paragraph: {paragraph_count}, page: {page_count})")
    print(f"  Skipped files: {skipped}")
    print(f"  JSON: {json_path}")
    print(f"  Images: {images_dir}")
    print(f"\nNext steps:")
    print(f"  1. Ensure {dataset_name}.json and {dataset_name}/ are in LlamaFactory/data/")
    print(f"  2. Add to data/dataset_info.json:")
    print(f'     "{dataset_name}": {{')
    print(f'       "file_name": "{dataset_name}.json",')
    print(f'       "formatting": "sharegpt",')
    print(f'       "columns": {{ "messages": "messages", "images": "images" }},')
    print(f'       "tags": {{ "role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant" }}')
    print(f"     }}")
    print(f"  3. Use template: lighton_ocr")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PAGE-XML/ALTO-XML to ShareGPT format for LightOnOCR-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_dir", required=True, help="Input directory with XML + images")
    parser.add_argument(
        "--output_dir",
        default="./data",
        help="Output directory (default: ./data). JSON and image folder go here.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Dataset name (folder + JSON name). Default: derived from input_dir.",
    )
    parser.add_argument(
        "--format",
        choices=["pagexml", "alto", "auto"],
        default="auto",
        help="XML format (default: auto-detect)",
    )
    parser.add_argument(
        "--unicode_form",
        choices=["NFC", "NFD", "NFKC", "NFKD"],
        default="NFC",
        help="Unicode normalization (default: NFC)",
    )
    parser.add_argument("--min_text_length", type=int, default=1)
    parser.add_argument("--min_crop_size", type=int, default=32)
    parser.add_argument("--include_full_pages", action="store_true", default=True)
    parser.add_argument("--no_full_pages", action="store_false", dest="include_full_pages")
    parser.add_argument("--include_paragraphs", action="store_true", default=True)
    parser.add_argument("--no_paragraphs", action="store_false", dest="include_paragraphs")
    parser.add_argument("--paragraph_min_lines", type=int, default=5)
    parser.add_argument("--paragraph_max_lines", type=int, default=10)
    parser.add_argument("--line_separator", default="\n")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Files per batch for memory efficiency (default: 100)",
    )
    parser.add_argument("--image_format", default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument(
        "--symlink_images",
        action="store_true",
        help="Symlink full-page images instead of copying (saves space if source is local)",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Alternative directory for images (mirrors input_dir structure). "
        "Use when images are not co-located with XML files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print reason for each skipped file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only first N XML files (for debugging).",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or Path(args.input_dir).name
    if not dataset_name:
        dataset_name = "lightonocr_dataset"

    convert_to_sharegpt(
        args.input_dir,
        args.output_dir,
        dataset_name,
        format_type=args.format,
        unicode_form=args.unicode_form,
        min_text_length=args.min_text_length,
        min_crop_size=args.min_crop_size,
        include_full_pages=args.include_full_pages,
        include_paragraphs=args.include_paragraphs,
        paragraph_min_lines=args.paragraph_min_lines,
        paragraph_max_lines=args.paragraph_max_lines,
        line_separator=args.line_separator,
        batch_size=args.batch_size,
        image_format=args.image_format,
        symlink_images=args.symlink_images,
        image_dir=args.image_dir,
        verbose=args.verbose,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
