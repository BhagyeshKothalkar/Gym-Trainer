from io import BytesIO
from typing import Any, Dict

import cloudinary
import cloudinary.uploader

from config import CONFIG


class CloudinaryClient:
    def __init__(self):
        cloudinary.config(
            cloud_name=CONFIG.cloudinary.cloud_name,
            api_key=CONFIG.cloudinary.api_key,
            api_secret=CONFIG.cloudinary.api_secret,
            secure=True,
        )

    def upload_image_bytes(
        self, image_bytes: bytes, public_id: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        context = "|".join(
            f"{key}={value}" for key, value in metadata.items() if value is not None
        )
        result = cloudinary.uploader.upload(
            BytesIO(image_bytes),
            folder=CONFIG.cloudinary.folder,
            public_id=public_id,
            resource_type="image",
            overwrite=True,
            context=context,
        )
        return {
            "url": result["secure_url"],
            "public_id": result["public_id"],
            "metadata": metadata,
        }
