"""
Grad-CAM inference utility for the FastAPI backend.
"""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Hooks into the last conv block (features[-1]) to capture activations + gradients.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a (224, 224) float32 heatmap in [0, 1]."""
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[0, class_idx]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)


def _overlay_heatmap(image_tensor: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
    """Blend heatmap onto original image. Returns (224, 224, 3) uint8."""
    import matplotlib.pyplot as plt
    mean = np.array(_MEAN)
    std = np.array(_STD)
    img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean).clip(0, 1)

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]
    blended = (0.5 * img + 0.5 * colored).clip(0, 1)
    return (blended * 255).astype(np.uint8)


def generate_heatmap(image: Image.Image, model: torch.nn.Module, class_idx: int) -> str:
    """
    Generate a Grad-CAM heatmap overlay for the given image and predicted class.

    Returns a base64-encoded PNG string suitable for sending over HTTP.
    """
    model.eval()
    image_tensor = _preprocess(image.convert("RGB")).unsqueeze(0)

    gradcam = GradCAM(model)
    with torch.enable_grad():
        heatmap = gradcam.generate(image_tensor, class_idx)

    overlay = _overlay_heatmap(image_tensor, heatmap)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
