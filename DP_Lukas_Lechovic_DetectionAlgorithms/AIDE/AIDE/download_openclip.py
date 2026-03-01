"""
Download the correct OpenCLIP ConvNeXt XXL checkpoint for AIDE
"""
import os

def download_openclip_convnext():
    print("Downloading OpenCLIP ConvNeXt XXL checkpoint...")
    print("This is a large file (~3.5GB), it may take a while...")

    import open_clip

    # This will download the model automatically to cache
    model, _, _ = open_clip.create_model_and_transforms(
        "convnext_xxlarge",
        pretrained="laion2b_s34b_b82k_augreg_soup"
    )

    # Get the cache path
    import huggingface_hub
    cache_path = huggingface_hub.hf_hub_download(
        repo_id="laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup",
        filename="open_clip_pytorch_model.bin"
    )

    print(f"\nOpenCLIP ConvNeXt XXL downloaded!")
    print(f"Cache path: {cache_path}")

    # Copy to checkpoints folder
    import shutil
    os.makedirs('checkpoints', exist_ok=True)
    target_path = 'checkpoints/open_clip_pytorch_model.bin'
    shutil.copy(cache_path, target_path)
    print(f"Copied to: {target_path}")

if __name__ == '__main__':
    download_openclip_convnext()
