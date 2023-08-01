from typing import Literal

import cv2
import numpy as np
from invokeai.app.models.image import ImageCategory, ImageField, ResourceOrigin
from PIL import Image
from pydantic import Field

from .baseinvocation import BaseInvocation, InvocationConfig, InvocationContext
from .image import ImageOutput


class ExtrudeDepthInvocation(BaseInvocation):
    """Node for creating fake depth by "extruding" a mask using opencv."""

    # fmt: off
    type: Literal["cv_extrude_depth"] = "cv_extrude_depth"

    # Inputs
    mask: ImageField = Field(None, description="The mask from which to extrude")
    direction: float = Field(45., description="Extrude direction in degrees")
    shift: int = Field(40, description="Number of pixels to shift bottom from top")
    close_point: int = Field(180, ge=0, le=255, description="Closest extrusion depth")
    far_point: int = Field(80, ge=0, le=255, description="Farthest extrusion depth")
    bg_threshold: int = Field(10, ge=0, lt=255, description="Background threshold")
    bg_depth: int = Field(0, ge=0, lt=255, description="Target background depth")
    steps: int = Field(100, description="Number of steps in extrusion gradient")
    invert: bool = Field(False, description="Inverts mask image before extruding")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["cv", "mask", "depth"],
            },
        }

    def extrude(self, cv_mask: np.ndarray) -> np.ndarray:
        alpha = np.cos(self.direction / (2 * np.pi))
        beta = np.sin(self.direction / (2 * np.pi))
        canvas = np.zeros_like(cv_mask, dtype=np.float32)
        slope = (self.close_point - self.far_point) / 255
        offset = self.far_point

        for frac in np.linspace(0, 1, self.steps):
            shift = (1 - frac) * self.shift
            x_shift = np.round(alpha * shift).astype(int)
            y_shift = np.round(beta * shift).astype(int)
            canv_x = np.max((x_shift, 0))
            canv_y = np.max((y_shift, 0))
            mask_x = np.max((-x_shift, 0))
            mask_y = np.max((-y_shift, 0))
            mask_area = cv_mask[mask_x : -canv_x - 1, mask_y : -canv_y - 1]
            canvas_area = canvas[canv_x : -mask_x - 1, canv_y : -mask_y - 1]
            values = offset + slope * frac * mask_area
            sub_mask = values > canvas_area
            sub_mask &= self.bg_threshold < mask_area
            canvas_area[sub_mask] = values[sub_mask]
        canvas[canvas < self.bg_threshold] = self.bg_depth
        return canvas.astype(np.uint8)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        mask = context.services.images.get_pil_image(self.mask.image_name)

        # Convert to numpy and only one channel
        cv_mask = np.array(mask)
        if cv_mask.ndim == 3 and cv_mask.shape[-1] == 4:
            # Use alpha channel as mask - Does this make sense in general?
            cv_mask = cv_mask[:, :, -1]
            # invert uint8 (0, 1, ..., 255 -> 255, 254, ..., 0)
            cv_mask = (-1 - cv_mask).astype(np.uint8)
        elif cv_mask.ndim == 3:
            # RGB -> Grayscale
            cv_mask = cv2.cvtColor(cv_mask, cv2.COLOR_RGB2GRAY)

        if self.invert:
            # invert uint8 (0, 1, ..., 255 -> 255, 254, ..., 0)
            cv_mask = (-1 - cv_mask).astype(np.uint8)

        # Extrude
        cv_extruded = self.extrude(cv_mask)

        # Convert back to Pillow
        pil_extruded = Image.fromarray(cv2.cvtColor(cv_extruded, cv2.COLOR_GRAY2RGB))

        mask_dto = context.services.images.create(
            image=pil_extruded,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=mask_dto.image_name),
            width=mask_dto.width,
            height=mask_dto.height,
        )
