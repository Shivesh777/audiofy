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

from scipy.io import wavfile
import pyroomacoustics as pra


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


def stitch(audio_files, coordinates, room_dimensions, output_path):
    """
    Create spatial audio by simulating sound propagation in a virtual room.

    Parameters:
    - audio_files (np.array): Array of audio data.
    - coordinates (np.array): Array of 3D coordinates corresponding to the positions of audio data.
    - room_dimensions (list): List containing the dimensions of the virtual room in meters [length, width, height].
    - output_path (str): File path where the spatial audio will be saved.

    Returns:
    None

    Note:
    - The function assumes all audio files have the same sampling rate.
    - The listener's position (receiver) is set at the center of the room's ground level.
    - The simulated spatial audio is saved to the specified output path.

    """
    # Create an empty room
    room = pra.ShoeBox(room_dimensions)
    print(coordinates)

    # Place sources in the room
    for i, coord in enumerate(coordinates):
        audio = audio_files[i][0][0]
        print(audio)

        my_source = pra.SoundSource(coord, signal=audio)
        room.add_source(my_source)

    # Set receiver position (listener's position)
    receiver_pos = [room_dimensions[0] / 2, room_dimensions[1] / 2, 0]
    room.add_microphone(receiver_pos)

    # Compute RIRs
    room.compute_rir()

    # Simulate sound propagation in the room
    room.simulate()

    # Retrieve the simulated spatial audio signal received by the microphone
    print(room.mic_array.signals)
    output_signal = room.mic_array.signals[0]

    # Save the spatial audio to a file
    wavfile.write(output_path, 16000, output_signal)


def infer(image_path, output_path):
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
    
    depth  = ((-1 * (depth - 255.0))*(.1*w/255.)).detach().cpu().numpy()
    points = torch.floor(torch.Tensor(points)).to(torch.int32).detach().cpu().numpy()
    depth_values = depth[points[:, 1], points[:, 0]]
    points = np.hstack((points, depth_values.reshape(-1, 1)))
    
    stitch(audio, points, [w, h , 3*w], output_path)

infer("images/tiger.jpg", "output.wav")