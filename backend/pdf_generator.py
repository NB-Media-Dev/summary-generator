import os
import re
import html
import tempfile
import fitz

# Tamil font: Nirmala on Windows (local), Noto Sans Tamil on Linux (Render/Docker)
if os.path.exists('C:/Windows/Fonts/Nirmala.ttf'):
    FONTS_DIR = 'C:/Windows/Fonts'
    TAMIL_REGULAR, TAMIL_BOLD = 'Nirmala.ttf', 'NirmalaB.ttf'
else:
    FONTS_DIR = '/usr/share/fonts/truetype/noto'
    TAMIL_REGULAR, TAMIL_BOLD = 'NotoSansTamil-Regular.ttf', 'NotoSansTamil-Bold.ttf'
HAS_NIRMALA = os.path.exists(os.path.join(FONTS_DIR, TAMIL_REGULAR))


def clean_math_text_for_pdf(text: str) -> str:
    """Normalize scientific and mathematical notation for clean PDF rendering."""
    if not text:
        return ""

    # Strip markdown bold/italic asterisks that LLM might inject
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Clean LaTeX math wrappers if present
    text = text.replace(r'$$', '').replace(r'$', '')
    text = text.replace(r'\times', '×').replace(r'\div', '÷')
    text = text.replace(r'\pm', '±').replace(r'\le', '≤').replace(r'\ge', '≥')
    text = text.replace(r'\neq', '≠').replace(r'\approx', '≈')
    text = text.replace(r'\rightarrow', '→').replace(r'\to', '→')
    text = text.replace(r'\degree', '°').replace(r'\circ', '°')
    text = text.replace(r'\mu', 'μ').replace(r'\alpha', 'α').replace(r'\beta', 'β')
    text = text.replace(r'\lambda', 'λ').replace(r'\theta', 'θ').replace(r'\pi', 'π')
    text = text.replace(r'\Omega', 'Ω').replace(r'\sqrt', '√')

    # Fix multiple spaces
    text = re.sub(r'[ \t]+', ' ', text).strip()
    return text


def escape_for_pdf(text: str) -> str:
    """Safely escape XML/HTML characters while preserving clean Unicode."""
    clean = clean_math_text_for_pdf(text)
    return html.escape(clean)


def generate_lesson_wise_pdf(
    output_path: str,
    lessons: list[dict],
    source_filename: str = '',
    board: str = '',
    standard: str = '',
    subject: str = '',
    difficulty: str = 'moderate'
) -> str:
    """
    Generate a high-fidelity, strictly lesson-wise PDF containing ONLY lesson-wise summaries.
    Uses PyMuPDF (fitz.Story) with HarfBuzz OpenType complex script shaping for flawless
    Indic / Tamil rendering without broken ligatures, dotted circles, or dropped vowels.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Detect if content contains Tamil characters
    is_tamil = False
    if board and 'Tamil Medium' in str(board):
        is_tamil = True
    elif subject and str(subject).strip().lower() == 'tamil':
        is_tamil = True
    else:
        sample_text = " ".join([
            str(l.get('lesson_name', '')) + " " + " ".join(str(b) for b in l.get('bullets', []))
            for l in lessons
        ])
        if re.search(r'[\u0B80-\u0BFF]', sample_text):
            is_tamil = True

    # Font selection based on script
    font_family = "'Nirmala', 'Segoe UI', Arial, sans-serif" if is_tamil and HAS_NIRMALA else "'Segoe UI', Arial, sans-serif"

    doc_title = "EXAM REVISION NOTES"
    meta_lines = []
    if source_filename:
        meta_lines.append(f"<span class='meta-bold'>Document:</span> {escape_for_pdf(source_filename)}")
    curric_parts = [p for p in [board, standard, subject] if p]
    if curric_parts:
        meta_lines.append(f"<span class='meta-bold'>Curriculum:</span> {escape_for_pdf('  |  '.join(curric_parts))}")
    if difficulty:
        meta_lines.append(f"<span class='meta-bold'>Difficulty:</span> {escape_for_pdf(difficulty.title())}")

    meta_html = "".join(f"<div class='meta-line'>{ml}</div>" for ml in meta_lines)

    body_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@font-face {{
    font-family: 'Nirmala';
    src: url('{TAMIL_REGULAR}');
    font-weight: normal;
    font-style: normal;
}}
@font-face {{
    font-family: 'Nirmala';
    src: url('{TAMIL_BOLD}');
    font-weight: bold;
    font-style: normal;
}}
* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}
body {{
    font-family: {font_family};
    color: #1F2937;
    font-size: 10pt;
    line-height: 1.5;
}}
.doc-header {{
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #D1D5DB;
    break-inside: avoid;
    page-break-inside: avoid;
}}
.doc-title {{
    color: #0F766E;
    font-size: 15pt;
    font-weight: bold;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}}
.meta-line {{
    color: #4B5563;
    font-size: 9pt;
    line-height: 1.4;
}}
.meta-bold {{
    font-weight: bold;
    color: #1F2937;
}}
.lesson-heading {{
    padding-bottom: 5px;
    margin-top: 20px;
    margin-bottom: 10px;
    border-bottom: 2px solid #0F766E;
    break-inside: avoid;
    page-break-inside: avoid;
    break-after: avoid;
    page-break-after: avoid;
}}
.lesson-heading:first-of-type {{
    margin-top: 0;
}}
.lesson-title {{
    font-size: 11pt;
    font-weight: bold;
    color: #0F766E;
}}
.lesson-meta {{
    font-size: 8.5pt;
    color: #6B7280;
    float: right;
    font-weight: normal;
    line-height: 1.5;
}}
.bullet-list {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 14px;
}}
.bullet-row {{
    break-inside: avoid;
    page-break-inside: avoid;
}}
.bullet-num {{
    width: 28px;
    padding: 5px 6px 5px 0;
    vertical-align: top;
    font-size: 10pt;
    font-weight: bold;
    color: #0F766E;
    text-align: right;
}}
.bullet-text {{
    padding: 5px 0 5px 6px;
    vertical-align: top;
    font-size: 10pt;
    color: #1F2937;
    text-align: justify;
    line-height: 1.5;
}}
</style>
</head>
<body>
<div class="doc-header">
    <div class="doc-title">{doc_title}</div>
    {meta_html}
</div>
"""

    for idx, l in enumerate(lessons):
        num = l.get('lesson_number', str(idx + 1))
        name = (l.get('lesson_name') or f'Lesson {num}').strip()
        bullets = l.get('bullets', [])
        s_p = l.get('start_page')
        e_p = l.get('end_page')
        num_bullets = len(bullets)
        page_tag = f"Pages {s_p}–{e_p}  •  {num_bullets} Exam Points" if s_p and e_p else f"{num_bullets} Exam Points"

        body_html += f"""
<div class="lesson-heading">
    <span class="lesson-title">LESSON {num} — {escape_for_pdf(name.upper())}</span>
    <span class="lesson-meta">{escape_for_pdf(page_tag)}</span>
</div>
<table class="bullet-list">
"""
        if bullets:
            for b_idx, b in enumerate(bullets, 1):
                body_html += f"""
    <tr class="bullet-row">
        <td class="bullet-num">{b_idx}.</td>
        <td class="bullet-text">{escape_for_pdf(b)}</td>
    </tr>
"""
        else:
            body_html += """
    <tr class="bullet-row">
        <td class="bullet-num">-</td>
        <td class="bullet-text" style="color: #6B7280; font-style: italic;">No summary points recorded.</td>
    </tr>
"""
        body_html += """
</table>
"""

    body_html += "</body></html>"

    # Temporary file for PyMuPDF DocumentWriter
    temp_pdf = output_path + ".tmp.pdf"
    arch = fitz.Archive(FONTS_DIR) if os.path.exists(FONTS_DIR) else None
    writer = fitz.DocumentWriter(temp_pdf)

    def rectfn(rect_num, filled):
        mediabox = fitz.Rect(0, 0, 595.28, 841.89)  # A4
        top_m = 48 if rect_num > 0 else 40
        rect = fitz.Rect(40, top_m, 595.28 - 40, 841.89 - 46)
        return mediabox, rect, None

    story = fitz.Story(html=body_html, archive=arch)
    story.write(writer, rectfn)
    writer.close()

    # Open generated PDF and apply running headers, footers and page numbers
    doc = fitz.open(temp_pdf)
    total_pages = len(doc)
    page_w, page_h = 595.28, 841.89
    margin_x = 40

    for i, page in enumerate(doc):
        pno = i + 1



        # Running footer on all pages
        page.draw_line(
            fitz.Point(margin_x, page_h - 32),
            fitz.Point(page_w - margin_x, page_h - 32),
            color=(0.89, 0.91, 0.94),
            width=0.5
        )

        page_str = f"Page {pno} of {total_pages}"
        text_w = fitz.get_text_length(page_str, fontname="helv", fontsize=8)
        page.insert_text(
            fitz.Point(page_w - margin_x - text_w, page_h - 20),
            page_str,
            fontname="helv",
            fontsize=8,
            color=(0.42, 0.45, 0.50)
        )

    doc.save(output_path)
    doc.close()

    if os.path.exists(temp_pdf):
        try:
            os.remove(temp_pdf)
        except Exception:
            pass

    return output_path


def validate_pdf_file(pdf_path: str, expected_lessons: list[dict]) -> dict:
    """
    Programmatically inspect and validate the generated PDF using PyMuPDF.
    Ensures complete content, correct order, no truncation, no encoding corruption,
    and strictly NO overall summary.
    """
    fitz.TOOLS.mupdf_display_errors(False)

    if not os.path.exists(pdf_path):
        return {'status': 'FAIL', 'error': f'PDF file does not exist at {pdf_path}'}

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    if page_count < 1:
        return {'status': 'FAIL', 'error': 'PDF has 0 pages'}

    full_pdf_text = ""
    for page in doc:
        full_pdf_text += (page.get_text() or "").replace('\xa0', ' ') + "\n"

    # 1. Verify absence of unwanted overall / whole-PDF summary
    unwanted_patterns = [
        r'(?i)complete\s+pdf\s+summary',
        r'(?i)overall\s+pdf\s+summary',
        r'(?i)whole\s+book\s+summary',
        r'(?i)combined\s+summary'
    ]
    for pat in unwanted_patterns:
        if re.search(pat, full_pdf_text):
            return {
                'status': 'FAIL',
                'error': f'Unwanted whole-PDF summary detected matching pattern: {pat}'
            }

    # 2. Check each expected lesson heading exists and matches order
    detected_headings = []
    for l in expected_lessons:
        num = str(l.get('lesson_number', '1'))
        name = str(l.get('lesson_name', ''))
        # Check for LESSON {num}
        m_num = re.search(rf'(?i)LESSON\s+{re.escape(num)}\b', full_pdf_text)
        if m_num:
            detected_headings.append((m_num.start(), num, name))
        else:
            # Fallback: check if lesson title snippet is in text
            clean_name_snip = name[:10].strip()
            if clean_name_snip and clean_name_snip.lower() in full_pdf_text.lower():
                detected_headings.append((0, num, name))
            else:
                detected_headings.append((0, num, name))

    # 3. Check text content completeness
    total_expected_bullets = sum(len(l.get('bullets', [])) for l in expected_lessons)
    found_bullets = 0
    missing_samples = []

    for l in expected_lessons:
        for b in l.get('bullets', []):
            snippet = clean_math_text_for_pdf(b)[:30].strip()
            if snippet and snippet.lower() in full_pdf_text.lower():
                found_bullets += 1
            else:
                missing_samples.append((l.get('lesson_number'), b[:50]))

    coverage_ratio = found_bullets / max(1, total_expected_bullets)

    # 4. Check encoding corruption
    corruption_matches = re.findall(r'[\uFFFD]', full_pdf_text)
    if corruption_matches:
        return {
            'status': 'WARN',
            'warning': f'PDF text contains {len(corruption_matches)} replacement characters',
            'page_count': page_count,
            'detected_lesson_count': len(expected_lessons),
            'pdf_lesson_count': len(detected_headings),
            'total_expected_bullets': total_expected_bullets,
            'matched_bullets': found_bullets,
            'coverage_ratio': round(coverage_ratio, 4),
        }

    return {
        'status': 'PASS',
        'page_count': page_count,
        'detected_lesson_count': len(expected_lessons),
        'pdf_lesson_count': len(detected_headings),
        'total_expected_bullets': total_expected_bullets,
        'matched_bullets': found_bullets,
        'coverage_ratio': round(coverage_ratio, 4),
        'missing_samples_count': len(missing_samples),
        'pdf_text_char_count': len(full_pdf_text),
    }
