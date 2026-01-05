# app.py
import os
from io import BytesIO
from pathlib import Path

import streamlit as st

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions


st.set_page_config(page_title="PDF → Markdown (Docling, offline)", layout="wide")
st.title("PDF → Markdown using Docling (offline/air-gapped)")

with st.sidebar:
    st.header("Offline settings")

    # Where you copied Docling model artifacts to on the air-gapped machine
    artifacts_path = st.text_input(
        "Docling artifacts_path (folder containing model subfolders)",
        value="/Formpipe/Docling_converter/models",
        help=(
            "This must be the *parent* folder that contains multiple model directories "
            "(not a single model subfolder)."
        ),
    )

    force_hf_offline = st.checkbox(
        "Force Hugging Face Hub offline mode (HF_HUB_OFFLINE=1)",
        value=True,
        help="Fail fast if something tries to download from the internet.",
    )

    st.divider()
    st.header("Conversion options")
    do_ocr = st.checkbox("Enable OCR (for scanned PDFs)", value=False)
    do_table_structure = st.checkbox("Detect table structure", value=True)
    max_pages = st.number_input("Max pages to convert", min_value=1, value=200, step=1)


# Apply offline env var early (Streamlit re-runs; setting each run is fine).
if force_hf_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    os.environ.pop("HF_HUB_OFFLINE", None)


def validate_artifacts_path(p: str) -> tuple[bool, str]:
    if not p:
        return False, "artifacts_path is empty."
    path = Path(p).expanduser().resolve()
    if not path.exists():
        return False, f"Path does not exist: {path}"
    if not path.is_dir():
        return False, f"Path is not a directory: {path}"

    # Heuristic: expect at least some subfolders/files in the artifacts directory
    # (structure can vary by Docling version, so keep this non-strict).
    entries = list(path.iterdir())
    if len(entries) == 0:
        return False, f"Directory is empty: {path}"
    return True, str(path)


@st.cache_resource(show_spinner=False)
def build_converter(
    do_ocr: bool,
    do_table_structure: bool,
    artifacts_path_resolved: str,
) -> DocumentConverter:
    # Pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    pipeline_options.artifacts_path = artifacts_path_resolved

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


ok, artifacts_path_resolved_or_err = validate_artifacts_path(artifacts_path)
if not ok:
    st.error(f"Offline models folder issue: {artifacts_path_resolved_or_err}")
    st.info(
        "Fix the path in the sidebar to point at the folder where you copied Docling models."
    )
    st.stop()

artifacts_path_resolved = artifacts_path_resolved_or_err

uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
if not uploaded:
    st.info("Upload a PDF to get Markdown output.")
    st.stop()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Input")
    st.write(f"**File:** {uploaded.name}")
    st.write(f"**Size:** {uploaded.size:,} bytes")
    st.write(f"**Artifacts path:** `{artifacts_path_resolved}`")
    st.write(f"**HF_HUB_OFFLINE:** `{os.environ.get('HF_HUB_OFFLINE', '0')}`")

with st.spinner("Converting with Docling…"):
    try:
        converter = build_converter(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            artifacts_path_resolved=artifacts_path_resolved,
        )

        pdf_bytes = uploaded.getvalue()
        buf = BytesIO(pdf_bytes)

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
