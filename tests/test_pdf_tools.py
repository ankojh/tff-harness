import json

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.pdf_tools import PdfTools


def create_pdf(path, page_texts, title="Fixture PDF"):
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(content)
    writer.add_metadata({"/Title": title, "/Author": "Harness Tests"})
    with path.open("wb") as stream:
        writer.write(stream)


@pytest.mark.asyncio
async def test_read_pdf_extracts_selected_pages_and_metadata(tmp_path):
    create_pdf(tmp_path / "sample.pdf", ["First page", "Second page"])
    tools = PdfTools(tmp_path)

    result = json.loads(
        await tools.execute(
            "read_pdf",
            {"path": "sample.pdf", "pages": [2]},
        )
    )

    assert result["ok"] is True
    assert result["page_count"] == 2
    assert result["pages"][0]["page"] == 2
    assert "Second page" in result["pages"][0]["text"]
    assert result["metadata"]["Title"] == "Fixture PDF"
    assert result["pages_truncated"] is False


@pytest.mark.asyncio
async def test_read_pdf_defaults_to_first_twenty_pages(tmp_path):
    create_pdf(tmp_path / "long.pdf", [f"Page {index}" for index in range(1, 22)])
    tools = PdfTools(tmp_path)

    result = json.loads(await tools.execute("read_pdf", {"path": "long.pdf"}))

    assert result["ok"] is True
    assert result["page_count"] == 21
    assert len(result["pages"]) == 20
    assert result["pages_truncated"] is True


@pytest.mark.asyncio
async def test_read_pdf_warns_for_image_only_page(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with (tmp_path / "blank.pdf").open("wb") as stream:
        writer.write(stream)
    tools = PdfTools(tmp_path)

    result = json.loads(await tools.execute("read_pdf", {"path": "blank.pdf"}))

    assert result["ok"] is True
    assert "no extractable text" in result["warnings"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../outside.pdf", "/tmp/outside.pdf"])
async def test_read_pdf_rejects_paths_outside_workspace(tmp_path, path):
    tools = PdfTools(tmp_path)

    result = json.loads(await tools.execute("read_pdf", {"path": path}))

    assert result["ok"] is False


@pytest.mark.asyncio
async def test_read_pdf_rejects_invalid_page_number(tmp_path):
    create_pdf(tmp_path / "sample.pdf", ["Only page"])
    tools = PdfTools(tmp_path)

    result = json.loads(
        await tools.execute("read_pdf", {"path": "sample.pdf", "pages": [2]})
    )

    assert result["ok"] is False
    assert "1-page document" in result["error"]


@pytest.mark.asyncio
async def test_read_pdf_returns_structured_error_for_invalid_pdf(tmp_path):
    (tmp_path / "broken.pdf").write_text("not a pdf")
    tools = PdfTools(tmp_path)

    result = json.loads(await tools.execute("read_pdf", {"path": "broken.pdf"}))

    assert result["ok"] is False
