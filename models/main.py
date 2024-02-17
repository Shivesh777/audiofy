import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from core.models.model_module_infer import model_module
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision.transforms import Compose
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def load_image2audio_model():
    model_load_paths = [
        "CoDi_encoders.pth",
        "CoDi_text_diffuser.pth",
        "CoDi_audio_diffuser_m.pth",
        "CoDi_video_diffuser_8frames.pth",
    ]
    inference_tester = model_module(
        data_dir="checkpoints/", pth=model_load_paths, fp16=False
    )
    if torch.cuda.is_available():
        inference_tester = inference_tester.cuda()
    inference_tester = inference_tester.eval()
    return inference_tester


def load_depth_model():
    encoder = "vitl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (
        DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14")
        .to(device)
        .eval()
    )


def load_segmentation_model():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def infer(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = load_segmentation_model()
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    masked = []
    points = []
    for i in range(len(masks)):
        binary_mask = masks[i]["segmentation"].astype(np.uint8) * 255
        if masks[i]["area"] < 40000:
            continue
        binary_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        segmented_image = cv2.bitwise_and(image, binary_mask_3ch)
        masked.append(segmented_image)
        points.append(
            np.array(
                [
                    masks[i]["bbox"][0] + masks[i]["bbox"][2] / 2,
                    masks[i]["bbox"][1] + masks[i]["bbox"][3] / 2,
                ]
            )
        )

    model = load_depth_model()
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    depth = model(image)
    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[
        0, 0
    ]
    raw_depth = Image.fromarray(depth.detach().cpu().numpy().astype("uint16"))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    raw_depth.save(tmp.name)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    inference_tester = load_image2audio_model()
    audio = []
    for object in masked:
        audio_wave = inference_tester.inference(
            xtype=["audio"],
            condition=[object],
            condition_types=["image"],
            scale=7.5,
            n_samples=1,
            ddim_steps=500,
        )[0]
        audio.append(audio_wave)
