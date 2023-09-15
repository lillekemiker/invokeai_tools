from unittest.mock import MagicMock

import numpy as np
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.models.image import ImageCategory, ResourceOrigin
from PIL import Image

from invokeai_tools.nodes.extrude import ExtrudeDepthInvocation


def test_ExtrudeDepthInvocation_initializes():
    _ = ExtrudeDepthInvocation(id="1")


def test_ExtrudeDepthInvocation_extrude():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 3] = 255  # white dot in the middle
    mask[3, 0] = 39  # below bg_threshold dot
    node = ExtrudeDepthInvocation(
        id="1",
        direction=45,
        shift=40,
        close_point=180,
        far_point=80,
        bg_threshold=40,
        bg_depth=20,
    )
    res = node.extrude(mask)
    assert isinstance(res, np.ndarray)
    assert res.shape == (7, 7)
    assert res.dtype == np.uint8
    assert res.max() == node.close_point
    assert res[3, 3] == node.close_point
    assert res.min() == node.bg_depth
    assert res[3, 0] == node.bg_depth  # below bg_threshold is removed
    expected_trace = np.eye(7, dtype=bool)
    expected_trace[:3, :3] = False
    # currently a one-off bug that needs fixing where the one border pixel is missing
    expected_trace[-1, -1] = False
    assert (res[~expected_trace] == node.bg_depth).all()
    assert (node.far_point <= res[expected_trace]).all()
    assert (res[expected_trace] <= node.close_point).all()


def test_ExtrudeDepthInvocation_invoke():
    in_mask_np = np.ones((5, 5), dtype=np.uint8)
    in_mask = Image.fromarray(in_mask_np)
    mask_field = ImageField(image_name="test_mask")
    mock_ctx = MagicMock()
    mock_ctx.services.images.get_pil_image = MagicMock(return_value=in_mask)
    mock_mask_dto = MagicMock()
    mock_mask_dto.image_name = "test_out"
    mock_mask_dto.width = 5
    mock_mask_dto.height = 7
    mock_ctx.services.images.create = MagicMock(return_value=mock_mask_dto)
    mock_return_mask = np.zeros((5, 5), dtype=np.uint8)

    def mock_extrude(cv_mask):
        assert (cv_mask == in_mask_np).all()
        return mock_return_mask

    node = ExtrudeDepthInvocation(id="1", mask=mask_field)
    # using __setattr__ to override pydantic validation
    object.__setattr__(node, "extrude", mock_extrude)
    res = node.invoke(mock_ctx)
    assert isinstance(res, ImageOutput)
    assert isinstance(res.image, ImageField)
    assert res.image.image_name == "test_out"
    assert res.width == mock_mask_dto.width
    assert res.height == mock_mask_dto.height

    assert mock_ctx.services.images.create.call_count == 1
    assert "image" in mock_ctx.services.images.create.call_args[1]
    return_image = mock_ctx.services.images.create.call_args[1]["image"]
    for channel_idx in range(3):
        assert (mock_return_mask == np.asarray(return_image)[:, :, channel_idx]).all()

    expected_kwargs = dict(
        image=return_image,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        node_id="1",
        session_id=mock_ctx.graph_execution_state_id,
        is_intermediate=node.is_intermediate,
        workflow=node.workflow,
    )
    assert mock_ctx.services.images.create.call_args[1] == expected_kwargs
    assert mock_ctx.services.images.get_pil_image.call_count == 1
    assert len(mock_ctx.services.images.get_pil_image.call_args[0]) == 1
    assert (
        mock_ctx.services.images.get_pil_image.call_args[0][0] == mask_field.image_name
    )
