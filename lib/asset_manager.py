from .utils import get_logger
import os
from dataclasses import dataclass
from typing import Dict
import json
import uuid
from fastapi import UploadFile
import shutil
from pyservicemaker.utils import MediaInfo

logger = get_logger(__name__)

DEFAULT_ASSET_DIR = "/tmp/assets"
@dataclass
class Asset:
    id: str
    file_name: str
    mime_type: str
    size: int
    duration: int
    path: str
    use_count: int
    asset_dir: str
    description: str
    username: str
    password: str

    def lock(self):
        self._use_count += 1

    def unlock(self):
        self._use_count -= 1

    @classmethod
    def fromdir(cls, asset_dir):
        with open(os.path.join(asset_dir, "info.json")) as f:
            info = json.load(f)
            try:
                size = os.path.getsize(os.path.join(asset_dir, info["fileName"]))
            except:
                size = 0
            return Asset(id=info["assetId"],
                         path=info["path"],
                         file_name=info["fileName"],
                         mime_type=info["mimeType"],
                         duration=info["duration"],
                         username=info["username"],
                         password=info["password"],
                         description=info["description"],
                         asset_dir=asset_dir,
                         use_count=0,
                         size=size)


class AssetManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AssetManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._asset_dir = DEFAULT_ASSET_DIR
        self._asset_map:Dict[str, Asset] = dict()
        os.makedirs(self._asset_dir, exist_ok=True)
        asset_ids = self._get_existing_asset_ids()
        self._asset_map: dict[str, Asset] = {
            asset_id: Asset.fromdir(os.path.join(self._asset_dir, asset_id))
            for asset_id in asset_ids
        }
        self._initialized = True

    def save_file(self, file: UploadFile, file_name: str, mime_type: str) -> Asset | None:
        asset_id = str(uuid.uuid4())
        while asset_id in self._asset_map:
            asset_id = str(uuid.uuid4())
        asset_dir = os.path.join(self._asset_dir, asset_id)
        try:
            os.makedirs(asset_dir)
        except:
            logger.error(f"Failed to create asset directory: {asset_dir}")
            return None

        with open(os.path.join(asset_dir, file_name), "wb") as f:
            shutil.copyfileobj(file, f)
        try:
            size = os.path.getsize(os.path.join(asset_dir, file_name))
        except:
            size = 0
        mediainfo = MediaInfo.discover(os.path.join(asset_dir, file_name))

        with open(os.path.join(asset_dir, "info.json"), "w") as f:
            json.dump(
                {
                    "assetId": asset_id,
                    "path": os.path.join(asset_dir, file_name),
                    "fileName": file_name,
                    "mimeType": mime_type,
                    "duration": mediainfo.duration,
                    "username": "",
                    "password": "",
                    "description": "",
                    "size": size,
                }, f)

        self._asset_map[asset_id] = Asset.fromdir(asset_dir)

        logger.info(f"Saved file - asset-id: {asset_id} name: {file_name}")

        return self._asset_map[asset_id]

    def list_assets(self):
        return list(self._asset_map.values())

    def get_asset(self, asset_id):
        if asset_id not in self._asset_map:
            return None
        return self._asset_map[asset_id]

    def delete_asset(self, asset_id):
        if asset_id not in self._asset_map:
            return False
        if self._asset_map[asset_id].use_count > 0:
            logger.error(f"Asset {asset_id} is still in use")
            return False
        asset_dir = os.path.join(self._asset_dir, asset_id)
        shutil.rmtree(asset_dir)
        self._asset_map.pop(asset_id)
        logger.info(f"Removed asset {asset_id} and cleaned up associated resources")
        return True

    def _get_existing_asset_ids(self):
        entries = os.listdir(self._asset_dir)
        return [
            entry for entry in entries if os.path.isdir(os.path.join(self._asset_dir, entry))
            and os.path.isfile(os.path.join(self._asset_dir, entry, "info.json"))
        ]
