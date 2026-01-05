import streamlit as st
from io import BytesIO

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions


st.set_page_config(page_title="PDF → Markdown (Docling)", layout="wide")
st.title("PDF → Markdown using Docling")

with st.sidebar:
    st.header("Conversion options")

    do_ocr = st.checkbox("Enable OCR (for scanned PDFs)", value=False)
    do_table_structure = st.checkbox("Detect table structure", value=True)
    max_pages = st.number_input("Max pages to convert", min_value=1, value=200, step=1)

    st.caption(
        "Docling can also prefetch models for offline use (see `docling-tools models download`)."
    )

uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

@st.cache_resource(show_spinner=False)
def build_converter(do_ocr: bool, do_table_structure: bool) -> DocumentConverter:
    # Use attribute assignment for version-robustness across Docling releases.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

if not uploaded:
    st.info("Upload a PDF to get Markdown output.")
    st.stop()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Input")
    st.write(f"**File:** {uploaded.name}")
    st.write(f"**Size:** {uploaded.size:,} bytes")

with st.spinner("Converting with Docling…"):
    try:
        converter = build_converter(do_ocr=do_ocr, do_table_structure=do_table_structure)

        pdf_bytes = uploaded.getvalue()
        buf = BytesIO(pdf_bytes)

        # Convert from binary stream (works well with Streamlit uploads).
        source = DocumentStream(name=uploaded.name, stream=buf)
        result = converter.convert(source, max_num_pages=int(max_pages))

        doc = result.document
        markdown = doc.export_to_markdown()

    except Exception as e:
        st.error("Conversion failed.")
        st.exception(e)
        st.stop()

base_name = uploaded.name.rsplit(".", 1)[0]
md_filename = f"{base_name}.md"

with col_right:
    st.subheader("Markdown output")

    st.download_button(
        label="Download Markdown (.md)",
        data=markdown,
        file_name=md_filename,
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("Preview (rendered)", expanded=False):
        st.markdown(markdown)

    st.text_area("Raw Markdown", value=markdown, height=520)
