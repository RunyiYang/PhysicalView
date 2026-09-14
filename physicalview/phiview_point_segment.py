"""SAM3 point prompting on an actual viewer frame, in the inference environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def choose_mask(masks, scores, x, y):
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores).reshape(-1)
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("Invalid segmentation output")
    h, w = masks.shape[1:]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Click outside image")
    valid = [
        i
        for i, mask in enumerate(masks)
        if mask[y, x]
        and 24 <= mask.sum() <= h * w * 0.5
        and np.isfinite(scores[i])
        and scores[i] >= 0.5
    ]
    if not valid:
        raise ValueError(
            "No confident object at this point. Try clicking nearer its center."
        )
    i = max(valid, key=lambda i: scores[i])
    return masks[i], float(scores[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--x", required=True, type=int)
    ap.add_argument("--y", required=True, type=int)
    ap.add_argument("--checkpoint", default=os.environ.get("PHIVIEW_SAM3_CHECKPOINT"))
    args = ap.parse_args()
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint = args.checkpoint
    if not checkpoint:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download("facebook/sam3", "sam3.pt", local_files_only=True)
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=checkpoint,
        load_from_HF=False,
        enable_inst_interactivity=True,
        compile=False,
    )
    processor = Sam3Processor(model, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(Image.open(args.image).convert("RGB"))
        masks, scores, _ = model.predict_inst(
            state,
            point_coords=np.array([[args.x, args.y]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
    mask, score = choose_mask(masks, scores, args.x, args.y)
    np.save(args.out / "mask.npy", mask)
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.out / "mask.png")
    report = {
        "method": "SAM3 positive point prompt",
        "score": score,
        "pixels": int(mask.sum()),
        "point_xy": [args.x, args.y],
        "checkpoint": str(checkpoint),
        "gpu": torch.cuda.get_device_name(),
        "fresh_inference": True,
    }
    (args.out / "segmentation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
