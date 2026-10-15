import re
from pathlib import Path, PureWindowsPath

from .ingestion_errors import AssetNotFoundError, InvalidAssetError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = PROJECT_ROOT / "assets"
_ASSET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def validate_asset_id(asset_id):
    if not isinstance(asset_id, str) or _ASSET_ID.fullmatch(asset_id) is None:
        raise InvalidAssetError("Invalid asset identifier")
    return asset_id


def asset_directory(asset_id, *, must_exist=False):
    validate_asset_id(asset_id)
    root = ASSETS_ROOT.resolve()
    directory = (root / asset_id).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise InvalidAssetError("Asset path is outside the asset root")
    if must_exist and not directory.is_dir():
        raise AssetNotFoundError("Asset not found")
    return directory


def asset_file(asset_id, name):
    directory = asset_directory(asset_id, must_exist=True)
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise InvalidAssetError("Asset file is outside its directory")
    return path


def source_filename(filename):
    if not isinstance(filename, str) or not filename:
        raise InvalidAssetError("A source filename is required")
    name = Path(PureWindowsPath(filename).name).name
    if name in {"", ".", ".."} or len(name) > 255 or any(ord(char) < 32 for char in name):
        raise InvalidAssetError("Invalid source filename")
    return name
