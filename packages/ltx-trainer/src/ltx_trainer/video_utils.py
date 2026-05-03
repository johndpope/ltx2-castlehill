"""Video I/O utilities using PyAV.
This module provides functions for reading and writing video files using PyAV,
with optional audio support.
"""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm


def get_video_frame_count(video_path: str | Path) -> int:
    """Get the number of frames in a video file.
    Args:
        video_path: Path to the video file
    Returns:
        Number of frames in the video
    """
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        frame_count = video_stream.frames
        if frame_count == 0:
            # Fallback: count frames by decoding
            frame_count = sum(1 for _ in container.decode(video=0))
    return frame_count


def read_video(video_path: str | Path, max_frames: int | None = None) -> tuple[Tensor, float]:
    """Load frames from a video file using PyAV.
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to read. If None, reads all frames.
    Returns:
        Video tensor with shape [F, C, H, W] in range [0, 1] and frames per second (fps).
    """
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate or video_stream.base_rate or 24)

        frames = []
        for frame in container.decode(video=0):
            if max_frames is not None and len(frames) >= max_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))

    frames_np = np.stack(frames, axis=0)  # [F, H, W, C]
    video = torch.from_numpy(frames_np).float().div(255.0)  # [F, H, W, C] in [0, 1]
    return video.permute(0, 3, 1, 2), fps  # [F, C, H, W]


def save_video(
    video_tensor: torch.Tensor,
    output_path: Path | str,
    fps: float = 24.0,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
) -> None:
    """Save a video tensor to a file using PyAV, optionally with audio.
    Args:
        video_tensor: Video tensor of shape [C, F, H, W] or [F, C, H, W] in range [0, 1] or [0, 255]
        output_path: Path to save the video
        fps: Frames per second for the output video
        audio: Optional audio tensor of shape [C, samples] or [samples, C] in range [-1, 1]
        audio_sample_rate: Sample rate for the audio (required if audio is provided)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to [F, H, W, C] uint8 numpy array
    video_np = _prepare_video_array(video_tensor)
    _, height, width, _ = video_np.shape

    with av.open(str(output_path), mode="w") as container:
        # Setup video stream
        video_stream = container.add_stream("libx264", rate=int(fps))
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": "18"}

        # Setup audio stream if needed
        if audio is not None:
            if audio_sample_rate is None:
                raise ValueError("audio_sample_rate must be provided when audio is given")
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = "stereo"
            audio_stream.time_base = Fraction(1, audio_sample_rate)

        # Write video frames
        for frame_array in video_np:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        # Write audio if provided
        if audio is not None:
            _write_audio(container, audio_stream, audio, audio_sample_rate)


def streaming_vae_decode_and_save(
    vae_decoder,
    all_latent: torch.Tensor,
    n_latent: int,
    actual_pixel: int,
    output_path: Path | str,
    fps: float,
    vae_device: str = "cuda:1",
    decode_batch: int = 8,
    overlap: int = 1,
    pixel_per_latent: int = 8,
    crf: int = 18,
) -> int:
    """Decode latents through VAE and stream-encode directly to mp4 without
    materializing the full pixel tensor in CPU RAM.

    For each overlap-batched VAE call: decode on `vae_device` → range-convert
    [-1,1] → [0,1] → uint8 on GPU → move to CPU → feed each frame into the
    libx264 encoder → drop. Peak CPU RAM cost is O(one batch of pixels).

    Args:
        vae_decoder: callable VAE returning [1, 3, F_pixel, H, W] in [-1, 1].
        all_latent: latent tensor [1, C, n_latent, H_lat, W_lat] (CPU or any device).
        n_latent: number of latent frames in `all_latent` along dim=2.
        actual_pixel: target pixel-frame count; encoding stops when reached.
        output_path: destination .mp4 (parents created if missing).
        fps: output framerate.
        vae_device: device the VAE lives on.
        decode_batch: latent frames per VAE call.
        overlap: latent-frame overlap between batches for temporal-VAE continuity.
        pixel_per_latent: temporal expansion factor (LTX-2 = 8).
        crf: libx264 CRF (lower = higher quality, larger file).

    Returns:
        Number of pixel frames written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stride = decode_batch - overlap

    # Probe first batch to learn H/W (avoids passing them in)
    with torch.inference_mode():
        probe_end = min(decode_batch, n_latent)
        probe_batch = all_latent[:, :, :probe_end].to(vae_device)
        probe_pixels = vae_decoder(probe_batch)  # [1, 3, F_pixel, H, W]
    height, width = probe_pixels.shape[3], probe_pixels.shape[4]
    # Convert + cast probe on GPU, ship to CPU as uint8
    probe_pixels = ((probe_pixels + 1.0) / 2.0).clamp(0, 1)
    probe_pixels = (probe_pixels * 255.0).to(torch.uint8)
    probe_np = probe_pixels[0].permute(1, 2, 3, 0).cpu().numpy()  # [F, H, W, 3]
    del probe_batch, probe_pixels
    torch.cuda.empty_cache()

    total_written = 0
    with av.open(str(output_path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=int(fps))
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": str(crf)}

        def _emit(frames_np: np.ndarray) -> None:
            nonlocal total_written
            for j in range(frames_np.shape[0]):
                if total_written >= actual_pixel:
                    return
                frame = av.VideoFrame.from_ndarray(frames_np[j], format="rgb24")
                for packet in video_stream.encode(frame):
                    container.mux(packet)
                total_written += 1

        # Emit probe batch (acts as i=0)
        _emit(probe_np)

        # Continue from stride onward; skip overlap region for non-first batches
        for i in tqdm(range(stride, n_latent, stride), desc="Streaming VAE decode", unit="batch"):
            if total_written >= actual_pixel:
                break
            end = min(i + decode_batch, n_latent)
            batch = all_latent[:, :, i:end].to(vae_device)
            with torch.inference_mode():
                pixels = vae_decoder(batch)  # [1, 3, F_pixel, H, W]
            pixels = ((pixels + 1.0) / 2.0).clamp(0, 1)
            pixels = (pixels * 255.0).to(torch.uint8)
            # Skip overlap region in pixel space
            skip = overlap * pixel_per_latent
            pixels = pixels[:, :, skip:]
            frames_np = pixels[0].permute(1, 2, 3, 0).cpu().numpy()
            _emit(frames_np)
            del batch, pixels, frames_np
            torch.cuda.empty_cache()

        # Flush encoder
        for packet in video_stream.encode():
            container.mux(packet)

    return total_written


def _prepare_video_array(video_tensor: torch.Tensor) -> np.ndarray:
    """Convert video tensor to [F, H, W, C] uint8 numpy array."""
    # Handle [C, F, H, W] vs [F, C, H, W] format
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, F, H, W] -> [F, C, H, W]

    # Normalize to [0, 255] uint8
    if video_tensor.max() <= 1.0:
        video_tensor = video_tensor * 255

    # [F, C, H, W] -> [F, H, W, C]
    return video_tensor.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    """Write audio tensor to container as stereo AAC."""
    audio = audio.cpu().float()

    # Normalize to [samples, 2] stereo format
    if audio.ndim == 1:
        audio = audio.unsqueeze(1).repeat(1, 2)  # Mono -> stereo
    elif audio.shape[0] == 2 and audio.shape[1] != 2:
        audio = audio.T  # [2, samples] -> [samples, 2]
    if audio.shape[1] == 1:
        audio = audio.repeat(1, 2)  # Mono -> stereo

    # Convert to int16 interleaved: [samples, 2] -> [1, samples*2]
    audio_int16 = (audio.clamp(-1, 1) * 32767).to(torch.int16)
    audio_interleaved = audio_int16.contiguous().view(1, -1).numpy()

    # Create audio frame
    frame = av.AudioFrame.from_ndarray(audio_interleaved, format="s16", layout="stereo")
    frame.sample_rate = sample_rate

    # Resample to encoder format and write
    resampler = av.audio.resampler.AudioResampler(
        format=audio_stream.codec_context.format,
        layout=audio_stream.codec_context.layout,
        rate=sample_rate,
    )

    pts = 0
    for resampled_frame in resampler.resample(frame):
        resampled_frame.pts = pts
        pts += resampled_frame.samples
        for packet in audio_stream.encode(resampled_frame):
            container.mux(packet)

    for packet in audio_stream.encode():
        container.mux(packet)
