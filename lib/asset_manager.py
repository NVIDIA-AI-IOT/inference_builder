# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

    def add_live_stream(self, url: str, description="", username="", password="") -> Asset | None:
        asset_id = str(uuid.uuid4())
        while asset_id in self._asset_map:
            asset_id = str(uuid.uuid4())
        asset_dir = os.path.join(self._asset_dir, asset_id)
        try:
            os.makedirs(asset_dir)
        except Exception as e:
            logger.error(f"Failed to create asset directory: {asset_dir}")
            return None

        with open(os.path.join(asset_dir, "info.json"), "w") as f:
            json.dump(
                {
                    "assetId": asset_id,
                    "path": url,
                    "fileName": "",
                    "mimeType": "",
                    "duration": 0,
                    "username": username,
                    "password": password,
                    "description": description,
                    "size": 0
                }, f)

        self._asset_map[asset_id] = Asset.fromdir(asset_dir)

        logger.info(f"Added live stream - asset-id: {asset_id} url: {url}")

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
