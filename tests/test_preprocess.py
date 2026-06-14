import pytest
from preprocess.preprocess import PDFPreprocessor

preprocessor = PDFPreprocessor()

# ── _split_sentences ──

def test_split_sentences_basic():
    text = "This is a sentence. This is the second sentence!"
    result = preprocessor._split_sentences(text)
    assert result == ["This is a sentence.", "This is the second sentence!"]


def test_split_sentences_protects_abbreviations():
    text = "For details see Fig. 2. Next sentence here."
    result = preprocessor._split_sentences(text)
    # The dot in the abbreviation "Fig." should not be treated as sentence end
    assert len(result) == 2
    assert "Fig. 2" in result[0]


def test_split_sentences_empty_input():
    assert preprocessor._split_sentences("") == []
    assert preprocessor._split_sentences("   ") == []


# ── _table_to_markdown ──

def test_table_to_markdown_valid_table():
    data = [["Column1", "Column2"], ["a", "b"], ["c", "d"]]
    md = preprocessor._table_to_markdown(data)
    assert "| Column1 | Column2 |" in md
    assert "| a | b |" in md
    assert "| c | d |" in md


def test_table_to_markdown_skips_sparse_table():
    # More than half of the cells are empty -> table should be rejected
    data = [["", ""], ["", ""], ["", "x"]]
    assert preprocessor._table_to_markdown(data) == ""


def test_table_to_markdown_skips_single_row():
    # At least 2 rows (header + data) required
    assert preprocessor._table_to_markdown([["only_header"]]) == ""


# ── _classify_line ──

def test_classify_line_numbered_heading():
    ltype, level = preprocessor._classify_line(
        "1. Introduction", median_size=14, bold_ratio=1.0, avg_size=10
    )
    assert ltype == "heading"
    assert level == 1


def test_classify_line_list_item():
    ltype, _ = preprocessor._classify_line(
        "- an item", median_size=10, bold_ratio=0.0, avg_size=10
    )
    assert ltype == "list_item"


def test_classify_line_normal_paragraph():
    ltype, _ = preprocessor._classify_line(
        "This is a long and normally sized paragraph line.",
        median_size=10, bold_ratio=0.0, avg_size=10,
    )
    assert ltype == "paragraph_line"


# ── _merge_para_lines ──

def test_merge_para_lines_combines_consecutive_paragraphs():
    elements = [
        {"type": "paragraph_line", "level": 0, "text": "First line."},
        {"type": "paragraph_line", "level": 0, "text": "Second line."},
        {"type": "heading", "level": 1, "text": "Title"},
    ]
    merged = preprocessor._merge_para_lines(elements)

    assert merged[0]["type"] == "paragraph"
    assert merged[0]["text"] == "First line. Second line."
    assert merged[1]["type"] == "heading"


# ── _build_sections ──

def test_build_sections_splits_by_heading():
    elements = [
        {"type": "heading", "level": 1, "text": "Section 1", "page": 1},
        {"type": "paragraph", "level": 0, "text": "Content 1", "page": 1},
        {"type": "heading", "level": 1, "text": "Section 2", "page": 2},
        {"type": "paragraph", "level": 0, "text": "Content 2", "page": 2},
    ]
    sections = preprocessor._build_sections(elements)

    assert len(sections) == 2
    assert sections[0]["heading"] == "Section 1"
    assert sections[1]["heading"] == "Section 2"


def test_build_sections_merges_consecutive_headings():
    elements = [
        {"type": "heading", "level": 1, "text": "Section 1", "page": 1},
        {"type": "heading", "level": 2, "text": "Subheading", "page": 1},
        {"type": "paragraph", "level": 0, "text": "Content", "page": 1},
    ]
    sections = preprocessor._build_sections(elements)

    assert len(sections) == 1
    assert sections[0]["heading"] == "Section 1 Subheading"


# ── _chunk_section ──

def test_chunk_section_fits_in_single_chunk():
    section = {
        "heading": "Title",
        "blocks": [{"text": "Short content.", "page": 1, "type": "paragraph"}],
        "pages": {1},
    }
    chunks = preprocessor._chunk_section(section, start_id=0)

    assert len(chunks) == 1
    assert "Title" in chunks[0]["text"]
    assert "Short content." in chunks[0]["text"]
    assert chunks[0]["pages"] == [1]


def test_chunk_section_empty_returns_empty_list():
    section = {"heading": "", "blocks": [], "pages": set()}
    assert preprocessor._chunk_section(section, start_id=0) == []

if __name__ == "__main__":
    pytest.main([__file__])