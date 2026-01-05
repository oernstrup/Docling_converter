# app.py
# Streamlit: PDF -> Markdown using Docling (offline/air-gapped)
# - Exports images in REFERENCED mode (Markdown + *_artifacts folder)
# - Optionally generates OFFLINE image captions with BLIP (CPU or GPU) and injects them as alt-text
# - Outputs a ZIP with Markdown + images (best for portability)

import os
import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import streamlit as st
from PIL import Image

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode


st.set_page_config(
    page_title="PDF → Markdown (Docling offline, referenced images + captions)",
    layout="wide",
)
st.title("PDF → Markdown using Docling (offline/air-gapped, REFERENCED images + descriptions)")


# -----------------------
# Helpers
# -----------------------
def set_attr_if_exists(obj, name: str, value):
    """Docling evolves; keep this app tolerant to missing attrs."""
    if hasattr(obj, name):
        setattr(obj, name, value)


def validate_dir(p: str) -> Tuple[bool, str]:
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


def force_offline_env(force: bool):
    """Hard offline mode (recommended in air-gapped setups)."""
    if force:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


@st.cache_resource(show_spinner=False)
def build_converter(
    do_ocr_: bool,
    do_table_structure_: bool,
    docling_artifacts_path_: str,
    export_images_: bool,
    include_page_images_: bool,
    images_scale_: float,
) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()

    # Core options
    set_attr_if_exists(pipeline_options, "do_ocr", do_ocr_)
    set_attr_if_exists(pipeline_options, "do_table_structure", do_table_structure_)
    set_attr_if_exists(pipeline_options, "artifacts_path", docling_artifacts_path_)

    # Image options (if present in your Docling version)
    set_attr_if_exists(pipeline_options, "images_scale", images_scale_)
    if export_images_:
        set_attr_if_exists(pipeline_options, "generate_picture_images", True)
        set_attr_if_exists(pipeline_options, "generate_page_images", bool(include_page_images_))
    else:
        set_attr_if_exists(pipeline_options, "generate_picture_images", False)
        set_attr_if_exists(pipeline_options, "generate_page_images", False)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


# -----------------------
# OFFLINE BLIP captioning (NO transformers.pipeline)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_blip_captioner(model_path: str, prefer_gpu: bool):
    """
    Load BLIP captioning model from LOCAL path.
    Uses AutoProcessor + BlipForConditionalGeneration (reliable offline).
    """
    import torch
    from transformers import AutoProcessor, BlipForConditionalGeneration

    device = "cpu"
    if prefer_gpu and torch.cuda.is_available():
        device = "cuda:0"

    # local_files_only=True ensures no downloads.
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    model.to(device)
    model.eval()

    return processor, model, device


def generate_blip_caption(processor, model, device: str, img_path: Path, max_new_tokens: int) -> str:
    import torch

    img = Image.open(img_path).convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    # Move tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens))

    caption = processor.decode(out_ids[0], skip_special_tokens=True)
    print("Generated caption: ", caption)
    return (caption or "").strip()


def rewrite_markdown_alt_text(
    markdown: str,
    artifacts_dir_name: str,
    captions_by_relpath: Dict[str, str],
    add_visible_caption_line: bool,
) -> str:
    """
    Replace:
      ![](doc_artifacts/figure_0001.png)
    with:
      ![caption](doc_artifacts/figure_0001.png)
    and optionally:
      *caption* on a new line.

    Only applies to image paths that start with '{artifacts_dir_name}/'.
    """
    img_pat = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")

    def esc_md(s: str) -> str:
        return s.replace("[", "\\[").replace("]", "\\]")

    def repl(m):
        path = m.group("path").strip().strip('"').strip("'")
        if not path.startswith(artifacts_dir_name + "/"):
            return m.group(0)

        cap = captions_by_relpath.get(path, "").strip()
        if not cap:
            return m.group(0)

        safe = esc_md(cap)
        new_img = f"![{safe}]({path})"
        if add_visible_caption_line:
            return new_img + f"\n\n*{safe}*\n"
        return new_img

    return img_pat.sub(repl, markdown)


# -----------------------
# UI
# -----------------------
with st.sidebar:
    st.header("Offline settings")
    docling_models_path = st.text_input(
        "Docling artifacts_path (folder containing model subfolders)",
        value=r"C:\Formpipe\Docling_converter\models",
        help="Point to the parent folder containing Docling model artifacts.",
    )

    hard_offline = st.checkbox(
        "Force offline mode (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)",
        value=True,
    )

    st.divider()
    st.header("Conversion options")
    do_ocr = st.checkbox("Enable OCR (for scanned PDFs)", value=False)
    do_table_structure = st.checkbox("Detect table structure", value=True)
    max_pages = st.number_input("Max pages to convert", min_value=1, value=200, step=1)

    st.divider()
    st.header("Images (REFERENCED)")
    export_images = st.checkbox("Export figure images", value=True)
    images_scale = st.slider("Image resolution scale", 1.0, 3.0, 2.0, 0.5)
    include_page_images = st.checkbox("Also generate page images (optional)", value=False)

    st.divider()
    st.header("Image descriptions (offline BLIP)")
    enable_captions = st.checkbox("Generate descriptions for extracted images", value=True)

    caption_model_path = st.text_input(
        "Local BLIP model path",
        value=r"C:\Formpipe\Docling_converter\blip-image-captioning-base",
        help="Local Transformers model folder (must contain config.json + weights + processor files).",
    )

    prefer_gpu = st.checkbox("Use GPU for captions (if available)", value=False)
    caption_max_tokens = st.number_input("Caption length (max_new_tokens)", 10, 120, 40, 5)
    add_visible_caption_line = st.checkbox("Also add visible caption line under each image", value=False)

# Apply offline env flags
force_offline_env(hard_offline)

# Validate Docling artifacts dir
ok, docling_models_path_resolved_or_err = validate_dir(docling_models_path)
if not ok:
    st.error(f"Docling artifacts_path issue: {docling_models_path_resolved_or_err}")
    st.stop()
docling_models_path_resolved = docling_models_path_resolved_or_err

uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
if not uploaded:
    st.info("Upload a PDF to get Markdown output.")
    st.stop()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Input")
    st.write(f"**File:** {uploaded.name}")
    st.write(f"**Size:** {uploaded.size:,} bytes")
    st.write(f"**Docling artifacts_path:** `{docling_models_path_resolved}`")
    st.write(f"**HF_HUB_OFFLINE:** `{os.environ.get('HF_HUB_OFFLINE', '0')}`")
    st.write(f"**TRANSFORMERS_OFFLINE:** `{os.environ.get('TRANSFORMERS_OFFLINE', '0')}`")
    st.write("**Markdown image mode:** `REFERENCED`")

# Optionally load BLIP captioner (offline)
processor = model = None
caption_device_label = "off"
if enable_captions:
    cap_ok, cap_path_resolved_or_err = validate_dir(caption_model_path)
    if not cap_ok:
        st.warning(f"Captions disabled: caption model path issue: {cap_path_resolved_or_err}")
        enable_captions = False
    else:
        try:
            processor, model, device = load_blip_captioner(cap_path_resolved_or_err, prefer_gpu=prefer_gpu)
            caption_device_label = device
        except Exception as e:
            st.warning("Captions disabled: failed to load BLIP model from local path.")
            st.exception(e)
            enable_captions = False

with col_left:
    st.write(f"**Captioning device:** `{caption_device_label}`")

# -----------------------
# Convert + Export
# -----------------------
with st.spinner("Converting with Docling…"):
    try:
        converter = build_converter(
            do_ocr_=do_ocr,
            do_table_structure_=do_table_structure,
            docling_artifacts_path_=docling_models_path_resolved,
            export_images_=export_images,
            include_page_images_=include_page_images,
            images_scale_=images_scale,
        )

        pdf_bytes = uploaded.getvalue()
        source = DocumentStream(name=uploaded.name, stream=BytesIO(pdf_bytes))

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

            # Export markdown + referenced images
            doc.save_as_markdown(
                md_path,
                artifacts_dir=artifacts_dir,
                image_mode=ImageRefMode.REFERENCED,
            )

            markdown = md_path.read_text(encoding="utf-8")

            # Caption exported images and rewrite markdown
            image_files = sorted([p for p in artifacts_dir.rglob("*") if p.is_file()])
            raster_images = [
                p for p in image_files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]

            captions_by_relpath: Dict[str, str] = {}
            if enable_captions and processor and model and raster_images:
                rel_artifacts_dir_name = artifacts_dir.name  # used in markdown links
                for img_path in raster_images:
                    rel_path = f"{rel_artifacts_dir_name}/{img_path.relative_to(artifacts_dir).as_posix()}"
                    try:
                        cap = generate_blip_caption(
                            processor, model, caption_device_label, img_path, max_new_tokens=int(caption_max_tokens)
                        )
                        if cap:
                            captions_by_relpath[rel_path] = cap
                    except Exception:
                        continue

                if captions_by_relpath:
                    markdown = rewrite_markdown_alt_text(
                        markdown=markdown,
                        artifacts_dir_name=artifacts_dir.name,
                        captions_by_relpath=captions_by_relpath,
                        add_visible_caption_line=add_visible_caption_line,
                    )
                    md_path.write_text(markdown, encoding="utf-8")

            # Zip md + artifacts (best for download)
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

            # Stats for UI
            num_imgs = len(raster_images)
            num_captioned = len(captions_by_relpath)

    except Exception as e:
        st.error("Conversion failed.")
        st.exception(e)
        st.stop()

# -----------------------
# Output UI
# -----------------------
with col_right:
    st.subheader("Output")
    base_name = uploaded.name.rsplit(".", 1)[0]

    st.download_button(
        label="Download ZIP (Markdown + images + captions)",
        data=zip_bytes,
        file_name=f"{base_name}_md_with_images_and_captions.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.download_button(
        label="Download Markdown only (.md) — links break without artifacts folder",
        data=markdown,
        file_name=f"{base_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.caption(f"Extracted images: {num_imgs} | Captioned: {num_captioned}")

    with st.expander("Preview (rendered)", expanded=False):
        st.markdown(markdown)

    st.text_area("Raw Markdown", value=markdown, height=420)
