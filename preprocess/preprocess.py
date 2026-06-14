import os
import re
import statistics
import pdfplumber
import fitz

import sys
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_IMAGE_SIZE, CACHE_DIR


class PDFPreprocessor:

    ABBREVIATIONS = [
        "e.g.", "i.e.", "Fig.", "fig.", "Dr.", "Mr.", "Mrs.",
        "vs.", "etc.", "al.", "no.", "No.", "vol.", "Vol.", "et.",
        "Eq.", "eq.", "Ref.", "ref.", "Sec.", "sec."
    ]

    MARGIN_RATIO_TOP = 0.05
    MARGIN_RATIO_BOTTOM = 0.05
    MARGIN_RATIO_LEFT = 0.04
    MARGIN_RATIO_RIGHT = 0.04

    SPACE_GAP_PT = 1.5 # gap threshold in points for inserting spaces between chars

    def __init__(self) -> None:
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.min_image_size = MIN_IMAGE_SIZE
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.CAPTION_PATTERN = re.compile(
            r'^(figure|fig\.?|şekil|tablo|table|chart|graph|grafik)\s*\d*\s*[:\-\.]',
            re.IGNORECASE
        )

    def __call__(self, pdf_path: str) -> tuple[list[dict], list[dict]]:
        """Returns (text_chunks, image_chunks)."""
        pages, table_chunks = self._extract_pages(pdf_path)
        text_chunks = self._chunk_structured(pages)
        text_chunks.extend(table_chunks)
        text_chunks = [c for c in text_chunks if len(c["text"]) >= 50 or c["type"] == "table"]
        image_chunks = self._extract_images(pdf_path)

        source = os.path.basename(pdf_path)
        stem = os.path.splitext(source)[0]
        for c in text_chunks + image_chunks:
            c["source_pdf"] = source
            c["chunk_id"] = f"{stem}_{c['chunk_id']}"

        return text_chunks, image_chunks

    # ── page extraction ──

    def _extract_pages(self, pdf_path: str) -> tuple[list[dict], list[dict]]:
        """Extract structured elements per page. Tables separated."""
        pages = []
        table_chunks = []
        tbl_id = 0

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.find_tables()
                table_bboxes = [t.bbox for t in tables]

                for t in tables:
                    md = self._table_to_markdown(t.extract())
                    if not md:
                        continue
                    cap = self._find_table_caption(page, t.bbox)
                    table_chunks.append({
                        "chunk_id": f"tbl_{tbl_id}",
                        "text": (f"{cap}\n\n{md}" if cap else md).strip(),
                        "pages": [i + 1], "type": "table"
                    })
                    tbl_id += 1

                elements = self._page_elements(page, table_bboxes)
                pages.append({"page": i + 1, "elements": elements})

        return pages, table_chunks

    def _page_elements(self, page, table_bboxes):
        """Build structured elements from page.chars: text + font info from same source."""
        pw, ph = page.width, page.height
        mt, mb = ph * self.MARGIN_RATIO_TOP, ph * self.MARGIN_RATIO_BOTTOM
        ml, mr = pw * self.MARGIN_RATIO_LEFT, pw * self.MARGIN_RATIO_RIGHT

        # get chars, filter margins and tables
        chars = [c for c in (page.chars or []) if c.get("text", "").strip()]
        chars = [c for c in chars
                 if c["top"] >= mt and c["bottom"] <= ph - mb
                 and c["x0"] >= ml and c["x1"] <= pw - mr
                 and not self._in_any_bbox(c, table_bboxes)]
        if not chars:
            return []

        all_sizes = [c["size"] for c in chars if c.get("size")]
        avg_size = statistics.median(all_sizes) if all_sizes else 10

        # group into lines, build text, get font metrics
        lines = self._chars_to_lines(chars)

        elements = []
        for ln in lines:
            text = ln["text"].strip()
            if not text:
                continue
            ltype, level = self._classify_line(text, ln["median_size"], ln["bold_ratio"], avg_size)
            elements.append({"type": ltype, "level": level, "text": text})

        return self._merge_para_lines(elements)

    def _in_any_bbox(self, c, bboxes):
        """Check if char is inside any bounding box."""
        for (x0, t, x1, b) in bboxes:
            if c["x0"] >= x0 - 2 and c["x1"] <= x1 + 2 and c["top"] >= t - 2 and c["bottom"] <= b + 2:
                return True
        return False

    def _chars_to_lines(self, chars):
        """Group chars by y-position, merge overlapping groups, build text + font metrics."""
        if not chars:
            return []
        groups = {}
        for c in chars:
            key = round(c["top"] / 2)
            groups.setdefault(key, []).append(c)

        merged = []
        for _, gc in sorted(groups.items()):
            gtop = min(c["top"] for c in gc)
            gbot = max(c["bottom"] for c in gc)
            if merged and gtop < merged[-1]["bot"]:
                merged[-1]["chars"].extend(gc)
                merged[-1]["bot"] = max(merged[-1]["bot"], gbot)
            else:
                merged.append({"chars": gc, "top": gtop, "bot": gbot})

        result = []
        for m in merged:
            lc = sorted(m["chars"], key=lambda c: c["x0"])
            # build text with absolute gap threshold
            text = ""
            prev_x1 = None
            for c in lc:
                ct = c.get("text", "")
                if not ct:
                    continue
                if prev_x1 is not None and c["x0"] - prev_x1 > self.SPACE_GAP_PT:
                    text += " "
                text += ct
                prev_x1 = c["x1"]

            sizes = [c["size"] for c in lc if c.get("size")]
            ms = statistics.median(sizes) if sizes else 10
            bc = sum(1 for c in lc if any(w in c.get("fontname", "").lower() for w in ["bold", "medi", "semi", "demi", "heav"]))
            br = bc / max(len(lc), 1)
            result.append({"text": text, "median_size": ms, "bold_ratio": br})
        return result

    def _classify_line(self, text, median_size, bold_ratio, avg_size):
        """Classify: heading / list_item / paragraph_line."""
        is_bold = bold_ratio > 0.5
        is_short = len(text) < 120
        is_larger = median_size > avg_size * 1.1

        # list items first
        if re.match(r'^[•\-–—\*]\s', text):
            return "list_item", 0
        if re.match(r'^\(\d+\)\s', text) or re.match(r'^\d+\)\s', text):
            return "list_item", 0

        # numbered heading (1, 1.1, 2.3.4) — needs bold/larger
        m = re.match(r'^(\d+(\.\d+){0,3})\.?\s+[A-ZÇŞĞÜÖİa-z]', text)
        if m and is_short and (is_larger or is_bold):
            return "heading", m.group(1).count('.') + 1
        if m and is_short:
            return "list_item", 0

        # unnumbered heading: must be BOTH larger AND (bold OR all-caps) + short
        if is_larger and is_short and len(text.split()) <= 10:
            is_allcaps = text == text.upper() and len(text) > 3
            if is_bold or is_allcaps:
                return "heading", 1

        return "paragraph_line", 0

    def _merge_para_lines(self, elements):
        """Merge consecutive paragraph_line into single paragraph."""
        merged = []
        buf = []
        for e in elements:
            if e["type"] == "paragraph_line":
                buf.append(e["text"])
            else:
                if buf:
                    merged.append({"type": "paragraph", "level": 0, "text": " ".join(buf)})
                    buf = []
                merged.append(e)
        if buf:
            merged.append({"type": "paragraph", "level": 0, "text": " ".join(buf)})
        return merged

    def _chunk_structured(self, pages):
        """Build sections from headings, chunk each section."""
        all_elems = []
        for p in pages:
            for e in p["elements"]:
                all_elems.append({**e, "page": p["page"]})
        if not all_elems:
            return []

        sections = self._build_sections(all_elems)
        chunks = []
        cid = 0
        for sec in sections:
            sc = self._chunk_section(sec, cid)
            chunks.extend(sc)
            cid += len(sc)
        return chunks

    def _build_sections(self, elements):
        """Split at headings. Consecutive headings without content merge into one."""
        sections = []
        cur = {"heading": "", "blocks": [], "pages": set()}
        for e in elements:
            if e["type"] == "heading":
                if cur["blocks"]:
                    # current section has content → save it, start new
                    sections.append(cur)
                    cur = {"heading": e["text"], "blocks": [], "pages": {e["page"]}}
                else:
                    # consecutive heading, no content yet → merge into current heading
                    cur["heading"] = (cur["heading"] + " " + e["text"]).strip() if cur["heading"] else e["text"]
                    cur["pages"].add(e["page"])
            else:
                cur["blocks"].append(e)
                cur["pages"].add(e["page"])
        if cur["blocks"] or cur["heading"]:
            sections.append(cur)
        return sections

    def _chunk_section(self, section, start_id):
        """Chunk section: paragraph-level fill, heading prefix, list grouping, paragraph overlap."""
        heading = section["heading"]
        blocks = self._group_lists(section["blocks"])
        if not blocks and not heading:
            return []

        prefix = f"{heading}\n" if heading else ""
        full = prefix + "\n".join(b["text"] for b in blocks)
        pages = sorted(section["pages"])

        # fits in one chunk
        if len(full) <= self.chunk_size:
            return [{"chunk_id": f"t_{start_id}", "text": full.strip(), "pages": pages, "type": "text"}]

        # paragraph-level chunking
        chunks = []
        i = 0
        while i < len(blocks):
            cur_text = prefix
            cur_pages = set()
            j = i
            while j < len(blocks) and len(cur_text) + len(blocks[j]["text"]) + 1 <= self.chunk_size:
                cur_text += blocks[j]["text"] + "\n"
                cur_pages.add(blocks[j].get("page", pages[0] if pages else 1))
                j += 1

            if j == i:
                # oversized single block — sentence split
                fb = self._split_big_block(blocks[i]["text"], prefix, pages, start_id + len(chunks))
                chunks.extend(fb)
                i += 1
                continue

            chunks.append({
                "chunk_id": f"t_{start_id + len(chunks)}",
                "text": cur_text.strip(), "pages": sorted(cur_pages), "type": "text"
            })
            i = j - 1 if j - i > 1 else j  # paragraph overlap
        return chunks

    def _split_big_block(self, text, prefix, pages, start_id):
        """Split oversized block by sentences."""
        sents = self._split_sentences(text)
        if not sents:
            return [{"chunk_id": f"t_{start_id}", "text": (prefix + text).strip(), "pages": pages, "type": "text"}]
        chunks = []
        cur = prefix
        for s in sents:
            if len(cur) + len(s) + 1 > self.chunk_size and cur.strip() != prefix.strip():
                chunks.append({"chunk_id": f"t_{start_id + len(chunks)}", "text": cur.strip(), "pages": pages, "type": "text"})
                cur = prefix
            cur += s + " "
        if cur.strip() and cur.strip() != prefix.strip():
            chunks.append({"chunk_id": f"t_{start_id + len(chunks)}", "text": cur.strip(), "pages": pages, "type": "text"})
        return chunks

    def _group_lists(self, elements):
        """Group consecutive list_items into single blocks."""
        blocks = []
        buf = []
        for e in elements:
            if e["type"] == "list_item":
                buf.append(e)
            else:
                if buf:
                    blocks.append({"text": "\n".join(b["text"] for b in buf),
                                   "page": buf[0].get("page", 0), "type": "list"})
                    buf = []
                blocks.append({"text": e["text"], "page": e.get("page", 0), "type": e["type"]})
        if buf:
            blocks.append({"text": "\n".join(b["text"] for b in buf),
                           "page": buf[0].get("page", 0), "type": "list"})
        return blocks

    def _split_sentences(self, text):
        """Split into sentences protecting abbreviations."""
        text = text.replace("\n", " ").strip()
        if not text:
            return []
        for a in self.ABBREVIATIONS:
            text = text.replace(a, a.replace(".", "##D##"))
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.replace("##D##", ".").strip() for s in sents if s.replace("##D##", ".").strip()]

    def _find_table_caption(self, page: pdfplumber.page.Page, table_bbox: tuple) -> Optional[str]:
        """Crop the areas around a table (top, bottom, left, right) and extract caption text if present."""
        x0, top, x1, bottom = table_bbox
        page_width = page.width
        page_height = page.height
        margin = 30
        offset = 5  # to avoid overlapping with table borders

        # regions = [
        #     (max(0, x0 - 20), max(0, top - margin), min(page_width, x1 + 20), top),                 # top
        #     (max(0, x0 - 20), bottom, min(page_width, x1 + 20), min(page_height, bottom + margin)), # bottom
        #     (max(0, x0 - margin), top, x0, bottom),                                                 # left
        #     (x1, top, min(page_width, x1 + margin), bottom)                                         # right
        # ]

        regions = [
            (offset, max(0, top - margin), page_width - offset, top),
            (offset, bottom, page_width - offset, min(page_height, bottom + margin)),
            (offset, top + offset, x0, bottom - offset),
            (x1, top + offset, page_width - offset, bottom - offset)
        ]

        for crop_x0, crop_y0, crop_x1, crop_y1 in regions:
            if crop_y0 >= crop_y1 or crop_x0 >= crop_x1:
                continue

            try:
                cropped = page.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                text = cropped.extract_text(x_tolerance=1, y_tolerance=3)
            except Exception:
                continue

            if not text or not text.strip():
                continue

            # Control caption pattern on each line of the extracted text
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            for i, line in enumerate(lines):
                if self.CAPTION_PATTERN.match(line):
                    # return " ".join(lines[i:])
                    return line  # Only return the matched caption line

        return None

    def _table_to_markdown(self, data):
        """Convert table to markdown. Skip sparse tables."""
        if not data or len(data) < 2:
            return ""
        cells = [c for r in data for c in r]
        if sum(1 for c in cells if not c or not str(c).strip()) / max(len(cells), 1) > 0.5:
            return ""
        clean = [[str(c).replace("\n", " ").strip() if c else "" for c in r] for r in data]
        ncols = max(len(r) for r in clean)
        clean = [(r + [""] * (ncols - len(r)))[:ncols] for r in clean]
        h = clean[0]
        md = ["| " + " | ".join(h) + " |", "|" + "|".join(["---"] * ncols) + "|"]
        for r in clean[1:]:
            md.append("| " + " | ".join(r) + " |")
        return "\n".join(md)

    def _extract_images(self, pdf_path):
        """Extract images via PyMuPDF, captions via pdfplumber."""
        chunks = []
        doc = fitz.open(pdf_path)
        plumber = pdfplumber.open(pdf_path)
        iid = 0
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            for pn in range(len(doc)):
                fp = doc[pn]
                pp = plumber.pages[pn]
                for info in fp.get_images(full=True):
                    xref = info[0]
                    try:
                        img = doc.extract_image(xref)
                    except Exception:
                        continue
                    if not img or img.get("width", 0) < self.min_image_size or img.get("height", 0) < self.min_image_size:
                        continue
                    path = os.path.join(self.cache_dir, f"{stem}_p{pn+1}_img{iid}.{img.get('ext','png')}")
                    with open(path, "wb") as f:
                        f.write(img["image"])
                    cap = self._image_caption(fp, pp, xref)
                    chunks.append({"chunk_id": f"img_{iid}", "image_path": path, "page": pn + 1,
                                   "pages": [pn + 1], "caption": cap, "description": None, "type": "image"})
                    iid += 1
        finally:
            plumber.close()
            doc.close()
        return chunks

    def _image_caption(self, fitz_page, plumber_page, xref) -> Optional[str]:
        """Extract caption below image."""
        try:
            rects = fitz_page.get_image_rects(xref)
            if not rects:
                return None
            r = rects[0]
        except Exception:
            return None
        ph, pw = plumber_page.height, plumber_page.width
        sx, sy = pw / fitz_page.rect.width, ph / fitz_page.rect.height
        cx0, cy0 = max(0, r.x0 * sx), r.y1 * sy
        cx1, cy1 = min(pw, r.x1 * sx), min(ph, cy0 + 30)
        if cy0 >= ph or cx0 >= cx1:
            return None
        try:
            text = plumber_page.crop((cx0, cy0, cx1, cy1)).extract_text(x_tolerance=1, y_tolerance=3)
        except Exception:
            return None
        if text and text.strip() and self.CAPTION_PATTERN.match(text.strip()):
            return text.strip()
        return None

if __name__ == "__main__":
    p = PDFPreprocessor()
    tc, ic = p("data/yolov8.pdf")
    print(f"{len(tc)} text chunks, {len(ic)} images\n")
    for c in tc:
        print(f"[{c['chunk_id']}] type={c['type']} pages={c['pages']} len={len(c['text'])}")
        print(f"  {c['text'][:200]}")
        print("---")

# if __name__ == "__main__":
#     preprocessor = PDFPreprocessor()
#     text_chunks, image_chunks = preprocessor("data/yolov8.pdf")
#     print(f"Extracted {len(text_chunks)} text chunks and {len(image_chunks)} image chunks")

#     for chunk in text_chunks:
#         if True: #chunk["pages"] == [7] or chunk["pages"] == [7,8]:
#             print(f"{chunk['text']}")
#             print("-----")
#             if "tbl" in chunk['chunk_id']:
#                 print(f"{chunk['text']}")
#                 print("-----")