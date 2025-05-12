import os
import subprocess

def ffmpeg_extract_subclip_no_data(video_path, start_time, end_time, output_file):
    """
    Extracts a subclip from video_path between start_time and end_time (in seconds)
    and writes it to output_file, disabling data streams.
    """
    duration = end_time - start_time
    cmd = [
        "ffmpeg",
        "-y",                   # overwrite output file if exists
        "-ss", str(start_time), # start time
        "-i", video_path,       # input file
        "-t", str(duration),    # duration of the clip
        "-map", "0",            # include all streams...
        "-dn",                  # ...but disable data streams
        "-c", "copy",           # copy codecs (no re-encoding)
        output_file
    ]
    subprocess.run(cmd, check=True)

def create_clips(video_path, start_times, end_times, output_folder):
    """
    Create video clips from a source video given lists of start and end times.
    Returns a list of created clip file paths.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    created_files = []
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        output_file = os.path.join(output_folder, f"clip_{i+1}.mp4")
        ffmpeg_extract_subclip_no_data(video_path, start, end, output_file)
        created_files.append(output_file)
    return created_files
