import shutil
from pathlib import Path

from s3fs import S3FileSystem

import rl.utils.io as io

BUCKET_NAME = io.getenv("RL_BUCKET_NAME")
BUCKET_PUBLIC_URL = io.getenv("RL_BUCKET_PUBLIC_URL")
BUCKET_ACCESS_KEY_ID = io.getenv("RL_BUCKET_ACCESS_KEY_ID")
BUCKET_SECRET_ACCESS_KEY = io.getenv("RL_BUCKET_SECRET_ACCESS_KEY")
BUCKET_ENDPOINT = io.getenv("RL_BUCKET_ENDPOINT")
BUCKET_REGION = io.getenv("RL_BUCKET_REGION")


def get_bucket_fs() -> S3FileSystem:
    if any(
        v is None
        for v in (
            BUCKET_ACCESS_KEY_ID,
            BUCKET_SECRET_ACCESS_KEY,
            BUCKET_ENDPOINT,
            BUCKET_REGION,
        )
    ):
        raise ValueError("Missing bucket credentials")
    return S3FileSystem(
        key=BUCKET_ACCESS_KEY_ID,
        secret=BUCKET_SECRET_ACCESS_KEY,
        endpoint_url=BUCKET_ENDPOINT,
        client_kwargs={"region_name": BUCKET_REGION},
    )


def get_full_s3_path(path: str):
    return f"{BUCKET_NAME}/{path}"


def write_file(input_path: Path, s3_path: str, fs: S3FileSystem) -> str:
    with (
        fs.open(get_full_s3_path(s3_path), "wb") as f,
        input_path.open("rb") as input_file,
    ):
        shutil.copyfileobj(input_file, f)
    return get_public_url(s3_path)


def read_file(s3_path: str, fs: S3FileSystem) -> bytes:
    with fs.open(get_full_s3_path(s3_path), "rb") as f:
        return f.read()


def get_public_url(s3_path: str):
    return f"{BUCKET_PUBLIC_URL}/{s3_path}"


def list_bucket_files(prefix: str, fs: S3FileSystem) -> set[str]:
    """Returns set of full paths for all files under the given prefix."""
    return {
        path.split(f"{BUCKET_NAME}/", 1)[1]  # Convert full s3 path to relative path
        for path in fs.glob(f"{BUCKET_NAME}/{prefix}/**")
        if not path.endswith("/")  # Skip directories
    }
