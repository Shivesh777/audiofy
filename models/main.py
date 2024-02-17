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
import subprocess

from scipy.io import wavfile
import pyroomacoustics as pra

def crop_and_fill(image_array, fill_image):
    non_zero_mask = np.any(image_array > 0, axis=-1)
    rows_nonzero, cols_nonzero = np.nonzero(non_zero_mask)
    row_min = np.min(rows_nonzero)
    row_max = np.max(rows_nonzero)
    col_min = np.min(cols_nonzero)
    col_max = np.max(cols_nonzero)
    cropped_array = image_array[row_min:row_max+1, col_min:col_max+1, :]
    black_mask = np.all(cropped_array == 0, axis=-1)
    cropped_filled = np.where(black_mask[:, :, None], fill_image[row_min:row_max+1, col_min:col_max+1, :], cropped_array)
    return cropped_filled

def load_image2audio_model():
    model_load_paths = [
        "CoDi_encoders.pth",
        "CoDi_text_diffuser.pth",
        "CoDi_audio_diffuser_m.pth",
        "CoDi_video_diffuser_8frames.pth",
    ]
    codi = model_module(
        data_dir="checkpoints/", pth=model_load_paths, fp16=False
    )
    if torch.cuda.is_available():
        codi = codi.cuda()
    codi = codi.eval()
    return codi


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


def gen_audio(audio_data, coordinates, room_dimensions, mic_coordinates, out_path):
    fs = 16000
    room = pra.ShoeBox(room_dimensions, fs=fs)

    # Place sources in the room
    for i, coord in enumerate(coordinates):
        audio = audio_data[i][0][0]
        
        my_source = pra.SoundSource(coord, signal=audio)
        room.add_source(my_source)

    # Set receiver position (listener's position)
    receiver_pos = [room_dimensions[0] / 2, room_dimensions[1] / 2, 0]
    mic_pos = [receiver_pos[i] + mic_coordinates[i] for i in range(3)]
    room.add_microphone(mic_pos)

    # Compute RIRs
    room.compute_rir()

    room.simulate()
    output_signal = room.mic_array.signals[0]

    # Save the spatial audio to a file
    wavfile.write(out_path, fs, output_signal)


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
    - If an error occurs during simulation, an error message is printed.

    """
    fs = 16000
    mic_coordinates = [
    [0, 0, 100],  # front_left
    [0, 0, -100],  # front_right
    [10, 0, 0],  # front_center
    [0, -100, 0],  # lfe
    [-100, 0, 0],  # back_left
    [0, 100, 0],  # back_right
    ]
    for i in range(len(mic_coordinates)):
        mic_coord = mic_coordinates[i]
        gen_audio(audio_files, coordinates, room_dimensions, mic_coord, f"speaker_{i}.wav")

    ffmpeg_command = [
        "ffmpeg", "-i", "speaker_0.wav", "-i", "speaker_1.wav",
        "-i", "speaker_2.wav", "-i", "speaker_3.wav",
        "-i", "speaker_4.wav", "-i", "speaker_5.wav",
        "-filter_complex", "[0:a][1:a][2:a][3:a][4:a][5:a]join=inputs=6:channel_layout=5.1:map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-BL|5.0-BR[a]; [a]volume=10.0[out]",
        "-map", "[out]", output_path
    ]

    subprocess.run(ffmpeg_command)


def infer(image_path, output_path, prompt):
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
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:3]
    masked = []
    points = []
    print(len(masks))
    for i in range(len(masks)):
        binary_mask = masks[i]["segmentation"].astype(np.uint8) * 255
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

    codi = load_image2audio_model()
    audio = []
    for object in masked:
        if prompt == "" or prompt is None:
            audio_wave = codi.inference(
                xtype=["audio"],
                condition=[crop_and_fill(object, cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))],
                condition_types=["image"],
                scale=7.5,
                n_samples=1,
                ddim_steps=500,
            )[0]
        else:
            audio_wave = codi.inference(
                xtype=["audio"],
                condition=[crop_and_fill(object, cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)), prompt],
                condition_types=["image", "text"],
                scale=7.5,
                n_samples=1,
                ddim_steps=500,
            )[0]
        audio.append(audio_wave)
    
    depth  = ((-1 * (depth - 255.0))*(.5*w/255.)).detach().cpu().numpy()
    points = torch.floor(torch.Tensor(points)).to(torch.int32).detach().cpu().numpy()
    depth_values = depth[points[:, 1], points[:, 0]]
    points = np.hstack((points, depth_values.reshape(-1, 1)))
    
    stitch(audio, points, [w, h , .5*w], output_path)