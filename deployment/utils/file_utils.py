import os
import zipfile
import io

def create_zip_for_clips(folder):
    """
    Zips all files in the given folder and returns a BytesIO buffer.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            zipf.write(file_path, arcname=filename)
    buffer.seek(0)
    return buffer
