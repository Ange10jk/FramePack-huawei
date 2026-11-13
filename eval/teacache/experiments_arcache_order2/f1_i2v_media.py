"""Validate MP4s and retime a compressed stream without changing its frames."""
from fractions import Fraction
import json
import subprocess

from f1_i2v_config import PARAMS


def ffmpeg(*args):
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *map(str, args)],
                          check=True, capture_output=True, text=True)


def probe(path, fps, width, height):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular video: {path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_frames", "-show_streams", "-show_format",
         "-of", "json", str(path)], check=True, capture_output=True, text=True)
    if result.stderr:
        raise ValueError(f"Decode error: {path}: {result.stderr}")
    info = json.loads(result.stdout)
    streams, container = info["streams"], info["format"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    stream, = streams
    frame_count = PARAMS["frames"]
    time_base = Fraction(stream["time_base"])
    duration = Fraction(frame_count, fps)
    pts = [int(frame["best_effort_timestamp"]) * time_base for frame in info["frames"]]
    if ("mp4" not in container["format_name"].split(",") or stream["codec_name"] != "h264"
            or stream["codec_type"] != "video" or stream["width"] != width or stream["height"] != height
            or int(stream["nb_read_frames"]) != frame_count
            or pts != [Fraction(index, fps) for index in range(frame_count)]
            or Fraction(stream["avg_frame_rate"]) != fps or Fraction(stream["r_frame_rate"]) != fps
            or int(stream["start_pts"]) != 0 or int(stream["duration_ts"]) * time_base != duration
            or abs(Fraction(container["duration"]) - duration) > Fraction(1, 1000)):
        raise ValueError(f"Unexpected video metadata: {path}: {stream}")
    return dict(width=width, height=height, frames=frame_count, fps=fps, duration=float(duration),
                bytes=path.stat().st_size)


def retime(raw, evaluation):
    import av

    if evaluation.exists() or evaluation.is_symlink():
        raise FileExistsError(evaluation)
    scale = Fraction(PARAMS["raw_fps"], PARAMS["eval_fps"])
    with av.open(str(raw)) as source, av.open(str(evaluation), "w", options={
        "movflags": "+faststart",
    }) as output:
        original, = source.streams.video
        stream = output.add_stream(template=original)
        stream.time_base = original.time_base * scale
        for packet in source.demux(original):
            if packet.dts is None:
                continue
            # A larger time base stretches PTS, DTS and duration together,
            # including the final frame and H.264's B-frame reorder offsets.
            packet.time_base *= scale
            packet.stream = stream
            output.mux(packet)


def frame_hashes(path):
    output = ffmpeg("-i", path, "-f", "framemd5", "-").stdout
    return [line.split(",")[-1].strip() for line in output.splitlines()
            if line and not line.startswith("#")]
