import os

import torch
import torch.nn as nn


class DINOv3Featurizer(nn.Module):
    """DINOv3 featurizer supporting three loading sources:

    - "torch_hub": loads via torch.hub.load('facebookresearch/dinov3', arch).
      Pass `weights` as a local .pth path or URL to override default weights.
    - "local": builds architecture via torch.hub (pretrained=False) then loads
      a local .pth state_dict from `weights`.
    - "huggingface": loads via transformers.AutoModel.from_pretrained(arch),
      where `arch` is a HuggingFace model ID (e.g.
      'facebook/dinov3-vits16plus-pretrain-lvd1689m') or a local directory
      containing a saved HuggingFace model.
    """

    def __init__(self, arch, patch_size, feat_type, weights=None, source="torch_hub"):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type
        self.source = source

        if source == "huggingface":
            self._load_huggingface(arch)
        elif source == "local":
            self._load_local(arch, weights)
        else:
            self._load_torch_hub(arch, weights)

    def _load_torch_hub(self, arch, weights):
        if weights is not None:
            self.model = torch.hub.load(
                'facebookresearch/dinov3', arch, weights=weights)
        else:
            self.model = torch.hub.load('facebookresearch/dinov3', arch)
        self.embed_dim = self.model.embed_dim

    def _load_local(self, arch, weights):
        """Build architecture without pretrained weights, then load state_dict."""
        if weights is None:
            raise ValueError("source='local' requires a weights path")
        self.model = torch.hub.load(
            'facebookresearch/dinov3', arch, pretrained=False)
        state_dict = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.embed_dim = self.model.embed_dim

    def _load_huggingface(self, model_id_or_path):
        """Load from HuggingFace Hub or a local HuggingFace-format directory."""
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id_or_path)
        self.embed_dim = self.model.config.hidden_size
        self.num_register_tokens = self.model.config.num_register_tokens

    def get_cls_token(self, img):
        if self.source == "huggingface":
            outputs = self.model(pixel_values=img)
            return outputs.last_hidden_state[:, 0]
        return self.model.forward(img)

    def forward(self, img, n=1, include_cls=False):
        h = img.shape[2] // self.patch_size
        w = img.shape[3] // self.patch_size

        if self.source == "huggingface":
            outputs = self.model(pixel_values=img)
            n_prefix = 1 + self.num_register_tokens
            patch_tokens = outputs.last_hidden_state[:, n_prefix:]
        else:
            output = self.model.forward_features(img)
            patch_tokens = output["x_norm_patchtokens"]

        return patch_tokens.reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
