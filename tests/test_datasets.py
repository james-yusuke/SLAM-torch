from pathlib import Path

import cv2
import numpy as np

from slam_torch.datasets import TartanAirSource


def test_tartanair_layout_and_frame_loading(tmp_path: Path) -> None:
    image_dir = tmp_path / "image_lcam_front"
    depth_dir = tmp_path / "depth_lcam_front"
    image_dir.mkdir()
    depth_dir.mkdir()
    image = np.full((24, 32, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(image_dir / "000000.png"), image)
    np.save(depth_dir / "000000_depth.npy", np.full((24, 32), 3.0, dtype=np.float32))
    (tmp_path / "pose_lcam_front.txt").write_text("0 0 0 0 0 0 1\n", encoding="utf-8")
    source = TartanAirSource(tmp_path)
    report = source.validate()
    assert report["valid"]
    frame = next(iter(source))
    assert frame.image.shape == (24, 32, 3)
    assert frame.ground_truth_depth is not None
    assert frame.ground_truth_pose is not None
