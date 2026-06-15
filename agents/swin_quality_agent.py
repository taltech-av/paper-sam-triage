import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

import config

# pipeline class_id → Swin output class index
PIPELINE_TO_SWIN = {
    2: 1,  # vehicle
    3: 2,  # sign
    4: 3,  # cyclist → human
    5: 3,  # pedestrian → human
}

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
LID_MEAN = [0.25258, 0.49281, 0.49566]
LID_STD  = [0.23424, 0.03092, 0.16651]

SWIN_SIZE = 384


class SwingQualityAgent:
    """
    Swin-based quality signal: runs once per frame, then answers per mask.

    Replaces the VLM quality agent. The Swin model was trained on ZOD good
    frames, so its per-pixel predictions tell us whether SAM mask pixels
    actually belong to the labeled class — far more reliable than a 7B VLM
    doing crop-level quality judgement.
    """

    VALID_OUTPUTS = ("good", "bad")
    SAFE_DEFAULT = "good"

    def __init__(self, threshold: float, device: str = "cuda:0"):
        self.threshold = threshold
        self.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> torch.nn.Module:
        import json

        # Both projects have a 'core' package. Temporarily remove the pipeline's
        # core from sys.modules so fusion-training's core can be imported cleanly.
        saved_core = {k: sys.modules.pop(k)
                      for k in list(sys.modules)
                      if k == "core" or k.startswith("core.")}

        if str(config.FUSION_DIR) not in sys.path:
            sys.path.insert(0, str(config.FUSION_DIR))

        try:
            from core.advanced_model_builder import AdvancedModelBuilder

            cfg_path  = config.SWIN_CFG_PATH
            ckpt_path = config.SWIN_CKPT_PATH

            with open(cfg_path) as f:
                cfg = json.load(f)

            builder = AdvancedModelBuilder(cfg, self.device)
            model = builder.build_model()
            model.to(self.device)

            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()
        finally:
            # Remove fusion-training's core entries, restore pipeline's
            for k in [k for k in sys.modules if k == "core" or k.startswith("core.")]:
                del sys.modules[k]
            sys.modules.update(saved_core)

        return model

    # ------------------------------------------------------------------
    # Frame-level inference (call once per frame, cache result in Bundle)
    # ------------------------------------------------------------------

    def predict_frame(
        self,
        camera_bgr: np.ndarray,
        lidar_bgr: np.ndarray,
    ) -> np.ndarray:
        """Return a [384, 384] uint8 array of Swin class predictions."""
        rgb = self._prep_rgb(camera_bgr)
        lid = self._prep_lidar(lidar_bgr)
        with torch.no_grad():
            _, logits = self.model(rgb, lid, modal="cross_fusion")
        return torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    def _prep_rgb(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (SWIN_SIZE, SWIN_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(RGB_MEAN, device=self.device).view(3, 1, 1)
        std  = torch.tensor(RGB_STD,  device=self.device).view(3, 1, 1)
        return ((t.to(self.device) - mean) / std).unsqueeze(0)

    def _prep_lidar(self, bgr: np.ndarray) -> torch.Tensor:
        # LiDAR PNG stores X/Y/Z in R/G/B channels — preserve channel order
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (SWIN_SIZE, SWIN_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(LID_MEAN, device=self.device).view(3, 1, 1)
        std  = torch.tensor(LID_STD,  device=self.device).view(3, 1, 1)
        return ((t.to(self.device) - mean) / std).unsqueeze(0)

    # ------------------------------------------------------------------
    # Mask-level scoring (called from triage_mask via bundle)
    # ------------------------------------------------------------------

    def agreement(self, bundle) -> float | None:
        """Return the fraction of mask pixels Swin predicts as the expected class."""
        pred   = bundle.swin_pred
        mask   = bundle.pixel_mask
        cls_id = bundle.metadata["class_id"]
        H      = bundle.metadata["image_height"]
        W      = bundle.metadata["image_width"]

        if pred is None or mask is None:
            return None

        swin_cls = PIPELINE_TO_SWIN.get(cls_id)
        if swin_cls is None:
            return None

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return 0.0

        swin_ys = (ys * SWIN_SIZE / H).astype(int).clip(0, SWIN_SIZE - 1)
        swin_xs = (xs * SWIN_SIZE / W).astype(int).clip(0, SWIN_SIZE - 1)
        return float(np.mean(pred[swin_ys, swin_xs] == swin_cls))

    def run(self, bundle) -> str:
        """Drop-in replacement for QualityAgent.run(bundle)."""
        score = self.agreement(bundle)
        if score is None:
            return self.SAFE_DEFAULT
        return "good" if score >= self.threshold else "bad"
