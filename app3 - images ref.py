import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode


st.set_page_config(page_title="PDF → Markdown (Docling, offline, referenced images)", layout="wide")
st.title("PDF → Markdown using Docling (offline/air-gapped, REFERENCED images)")


def set_attr_if_exists(obj, name: str, value):
    """Docling evolves; keep the app tolerant to missing attributes."""
    if hasattr(obj, name):
        setattr(obj, name, value)


def validate_dir(p: str) -> tuple[bool, str]:
    if not p:
        return False, "Path is empty."
    path = Path(p).expanduser().resolve()
    if not path.exists():
        return False, f"Path does not exist: {path}"
    if not path.is_dir():
        return False, f"Path is not a directory: {path}"
    if not any(path.iterdir()):
        return False, f"Directory is empty: {path}"
    return True, str(path)


with st.sidebar:
    st.header("Offline settings")
    artifacts_path = st.text_input(
        "Docling artifacts_path (folder containing model subfolders)",
        value="/Formpipe/Docling_converter/models",
        help="Must be the parent folder that contains the downloaded Docling model artifacts.",
    )

    force_hf_offline = st.checkbox(
        "Force HF offline mode (HF_HUB_OFFLINE=1)",
        value=True,
        help="Recommended in air-gapped setups to fail fast if anything tries to download.",
    )

    st.divider()
    st.header("Conversion options")
    do_ocr = st.checkbox("Enable OCR (for scanned PDFs)", value=False)
    do_table_structure = st.checkbox("Detect table structure", value=True)
    max_pages = st.number_input("Max pages to convert", min_value=1, value=200, step=1)

    st.divider()
    st.header("Image export (REFERENCED)")
    enable_images = st.checkbox("Export figure images", value=True)
    images_scale = st.slider(
        "Image resolution scale",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.5,
        help="Higher = sharper exported images (larger ZIP).",
    )
    include_page_images = st.checkbox(
        "Also generate page images (optional)",
        value=False,
        help="May create many images; mostly useful for debugging/visual review.",
    )


if force_hf_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    os.environ.pop("HF_HUB_OFFLINE", None)


ok, artifacts_path_resolved_or_err = validate_dir(artifacts_path)
if not ok:
    st.error(f"Offline models folder issue: {artifacts_path_resolved_or_err}")
    st.stop()
artifacts_path_resolved = artifacts_path_resolved_or_err


@st.cache_resource(show_spinner=False)
def build_converter(
    do_ocr_: bool,
    do_table_structure_: bool,
    artifacts_path_: str,
    enable_images_: bool,
    include_page_images_: bool,
    images_scale_: float,
) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    set_attr_if_exists(pipeline_options, "do_ocr", do_ocr_)
    set_attr_if_exists(pipeline_options, "do_table_structure", do_table_structure_)
    set_attr_if_exists(pipeline_options, "artifacts_path", artifacts_path_)

    # image-related (not all versions expose all attrs; we set if present)
    set_attr_if_exists(pipeline_options, "images_scale", images_scale_)
    if enable_images_:
        set_attr_if_exists(pipeline_options, "generate_picture_images", True)
        set_attr_if_exists(pipeline_options, "generate_page_images", bool(include_page_images_))
    else:
        set_attr_if_exists(pipeline_options, "generate_picture_images", False)
        set_attr_if_exists(pipeline_options, "generate_page_images", False)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
if not uploaded:
    st.info("Upload a PDF to get Markdown output.")
    st.stop()

image_mode = ImageRefMode.REFERENCED  # forced

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Input")
    st.write(f"**File:** {uploaded.name}")
    st.write(f"**Size:** {uploaded.size:,} bytes")
    st.write(f"**Artifacts path:** `{artifacts_path_resolved}`")
    st.write(f"**HF_HUB_OFFLINE:** `{os.environ.get('HF_HUB_OFFLINE', '0')}`")
    st.write("**Markdown image mode:** `REFERENCED`")


with st.spinner("Converting with Docling…"):
    try:
        converter = build_converter(
            do_ocr_=do_ocr,
            do_table_structure_=do_table_structure,
            artifacts_path_=artifacts_path_resolved,
            enable_images_=enable_images,
            include_page_images_=include_page_images,
            images_scale_=images_scale,
        )

        pdf_bytes = uploaded.getvalue()
        buf = BytesIO(pdf_bytes)
        source = DocumentStream(name=uploaded.name, stream=buf)

        conv_res = converter.convert(source, max_num_pages=int(max_pages))
        doc = conv_res.document

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            out_dir = tmpdir / "output"
            out_dir.mkdir(parents=True, exist_ok=True)

            base_name = uploaded.name.rsplit(".", 1)[0]
            md_path = out_dir / f"{base_name}.md"
            artifacts_dir = out_dir / f"{base_name}_artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Write markdown + artifacts directory
            doc.save_as_markdown(
                md_path,
                artifacts_dir=artifacts_dir,
                image_mode=image_mode,
            )

            markdown = md_path.read_text(encoding="utf-8")

            # Zip md + artifacts together (best for download / portability)
            image_files = sorted([p for p in artifacts_dir.rglob("*") if p.is_file()])
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(md_path, arcname=md_path.name)
                for p in image_files:
                    zf.write(
                        p,
                        arcname=str(Path(artifacts_dir.name) / p.relative_to(artifacts_dir)),
                    )
            zip_buf.seek(0)
            zip_bytes = zip_buf.read()

    except Exception as e:
        st.error("Conversion failed.")
        st.exception(e)
        st.stop()


with col_right:
    st.subheader("Output")

    base_name = uploaded.name.rsplit(".", 1)[0]

    st.download_button(
        label="Download ZIP (Markdown + images)",
        data=zip_bytes,
        file_name=f"{base_name}_md_with_images.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.download_button(
        label="Download Markdown only (.md) — links will break without artifacts folder",
        data=markdown,
        file_name=f"{base_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("Preview (rendered)", expanded=False):
        st.markdown(markdown)

    st.text_area("Raw Markdown", value=markdown, height=420)
