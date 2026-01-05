from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Salesforce/blip-image-captioning-base",
    local_dir=r"C:\Formpipe\Docling_converter\blip-image-captioning-base",
    local_dir_use_symlinks=False,
)
print("Done.")
