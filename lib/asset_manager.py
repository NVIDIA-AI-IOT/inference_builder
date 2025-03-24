from .utils import get_logger
import os
from dataclasses import dataclass
from typing import Dict
import json
import uuid
from fastapi import UploadFile
import shutil
logger = get_logger(__name__)

@dataclass
class Asset:
    id: str
    file_name: str
    mime_type: str
    purpose: str
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

            return Asset(id=info["assetId"],
                         path=info["path"],
                         file_ame=info["fileName"],
                         media_type=info["mediaType"],
                         codec=info["codec"],
                         purpose=info["purpose"],
                         username=info["username"],
                         password=info["password"],
                         description=info["description"],
                         asset_dir=asset_dir)


class AssetManager:
    def __init__(self, asset_dir: str):
        self._asset_dir = asset_dir
        self._asset_map = Dict[str, Asset] = dict()
        os.makedirs(self.asset_dir, exist_ok=True)

    def save_file(self, file: UploadFile, file_name: str, purpose: str, mime_type: str) -> str | None:
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
        asset_path = "file://" + os.path.abspath(os.path.join(asset_dir, file_name))

        with open(os.path.join(asset_dir, "info.json"), "w") as f:
            json.dump(
                {
                    "assetId": asset_id,
                    "path": os.path.join(asset_dir, file_name),
                    "fileName": file_name,
                    "mimeType": mime_type,
                    "purpose": purpose,
                    "username": "",
                    "password": "",
                    "description": "",
                }, f)

        self._asset_map[asset_id] = Asset.fromdir(asset_dir)

        logger.info(f"Saved file - asset-id: {asset_id} name: {file_name}")

        return asset_id

    def list_assets(self):
        return list(self._asset_map.values())

    def get_asset(self, asset_id):
        if asset_id not in self._asset_map:
            return None
        return self._asset_map[asset_id]

    def delete_asset(self, asset_id):
        if asset_id not in self._asset_map:
            return
        if self._asset_map[asset_id].use_count > 0:
            logger.error(f"Asset {asset_id} is still in use")
            return
        asset_dir = os.path.join(self._asset_dir, asset_id)
        shutil.rmtree(asset_dir)
        self._asset_map.pop(asset_id)
        logger.info(f"Removed asset {asset_id} and cleaned up associated resources")

    def _get_existing_asset_ids(self):
        entries = os.listdir(self._asset_dir)
        return [
            entry for entry in entries if os.path.isdir(os.path.join(self._asset_dir, entry))
            and os.path.isfile(os.path.join(self._asset_dir, entry, "info.json"))
        ]
