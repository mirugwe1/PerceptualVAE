"""
perceptual.py

Frozen feature extractors and the perceptual loss built on them.

A backbone returns a list of feature maps. PerceptualLoss never learns what
it is talking to, so adding one is a subclass plus a registry entry.

Each extractor owns three things that are silent-failure risks if handled
anywhere else: the channel count it expects, the normalization statistics its
weights were trained with, and any fixed input size. Putting them next to the
weights is the only arrangement where a mismatch is hard to introduce.

BACKBONES

  vgg19_imagenet   torchvision VGG-19. Natural photographs, so its early
                   layers respond to colour edges and object texture that a
                   radiograph does not contain. Useful mainly as the control.
  densenet_cxr     TorchXRayVision DenseNet-121 trained on chest radiographs
                   (NIH, CheXpert, MIMIC, PadChest, RSNA, or the combined
                   'all' model). Natively single channel, so no grayscale to
                   RGB repeat and no ImageNet statistics to get wrong. This
                   is the one I would use by default.
  resnet_cxr       TorchXRayVision ResNet-50 at 512 resolution.

NOTE ON COMPARING BACKBONES. Swapping VGG-19-ImageNet for DenseNet-CXR
changes architecture and pretraining domain at once. If the question is
whether the domain matters, compare densenet_imagenet against densenet_cxr
instead; both are available here for exactly that reason.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module, ABC):
    normalization: tuple[tuple[float, ...], tuple[float, ...]] = (
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    in_channels: int = 3
    input_size: int | None = None
    input_range: str = "unit"      # "unit" for [0,1] then normalize; "xrv" for [-1024,1024]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # A frozen extractor stays in eval mode even when the VAE around it
        # switches to training. Without this, any BatchNorm inside it would
        # keep updating its running statistics on chest X-rays and the loss
        # would change meaning as training proceeds.
        return super().train(False)

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != self.in_channels:
            raise ValueError(
                f"{type(self).__name__} expects {self.in_channels} channels, got {x.size(1)}")

        if self.input_size is not None and x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)

        if self.input_range == "xrv":
            # TorchXRayVision expects roughly [-1024, 1024] rather than [0, 1].
            return (x * 2.0 - 1.0) * 1024.0

        mean, std = self.normalization
        c = x.size(1)
        if len(mean) != c:
            raise ValueError(
                f"{type(self).__name__}: {len(mean)} normalization entries for "
                f"{c} channels. Statistics must match the weights.")
        m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, c, 1, 1)
        s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, c, 1, 1)
        return (x - m) / s

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        ...


# --------------------------------------------------------------------------- #

# Indices into torchvision vgg19.features. The "_1" names are the FIRST ReLU
# of each block and the "_2"/"_4" names the LAST, so a _1 selection sits one
# or more convolutions earlier at the same spatial scale and therefore carries
# less processed, more edge-like features.
VGG19_LAYERS = {
    "relu1_1": 1,  "relu1_2": 3,
    "relu2_1": 6,  "relu2_2": 8,
    "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
    "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
    "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
}


class VGG19ImageNet(FeatureExtractor):
    normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    in_channels = 3

    def __init__(self, layers=None, **_):
        super().__init__()
        from torchvision import models
        self.names = layers or ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]
        self.idx = [VGG19_LAYERS[n] for n in self.names]
        feats = models.vgg19(weights="IMAGENET1K_V1").features
        self.features = nn.Sequential(*list(feats[: max(self.idx) + 1]))
        self.freeze()

    def forward(self, x):
        h, want, out = self.prepare(x), set(self.idx), []
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in want:
                out.append(h)
        return out


class DenseNetCXR(FeatureExtractor):
    """TorchXRayVision DenseNet-121, trained on chest radiographs.

    Single channel natively. Hooks are placed after each dense block, which
    gives a clean four-scale hierarchy comparable to the VGG selection.
    """

    in_channels = 1
    input_size = 224          # the res224 weights expect this
    input_range = "xrv"

    def __init__(self, weights: str = "densenet121-res224-all", layers=None, **_):
        super().__init__()
        import torchxrayvision as xrv
        self.model = xrv.models.get_model(weights)
        self.names = layers or ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
        self.freeze()

    def forward(self, x):
        h = self.prepare(x)
        f = self.model.features
        h = f.pool0(f.relu0(f.norm0(f.conv0(h))))
        out = []
        for i in (1, 2, 3, 4):
            h = getattr(f, f"denseblock{i}")(h)
            if f"denseblock{i}" in self.names:
                out.append(h)
            if i < 4:
                h = getattr(f, f"transition{i}")(h)
        return out


class ResNetCXR(FeatureExtractor):
    """TorchXRayVision ResNet-50, 512 resolution, chest radiographs."""

    in_channels = 1
    input_size = 512
    input_range = "xrv"

    def __init__(self, weights: str = "resnet50-res512-all", layers=None, **_):
        super().__init__()
        import torchxrayvision as xrv
        self.model = xrv.models.get_model(weights)
        self.names = layers or ["layer1", "layer2", "layer3", "layer4"]
        self.freeze()

    def forward(self, x):
        h = self.prepare(x)
        m = self.model.model            # underlying torchvision resnet
        h = m.maxpool(m.relu(m.bn1(m.conv1(h))))
        out = []
        for name in ("layer1", "layer2", "layer3", "layer4"):
            h = getattr(m, name)(h)
            if name in self.names:
                out.append(h)
        return out


class DenseNetImageNet(FeatureExtractor):
    """torchvision DenseNet-121, ImageNet weights.

    Present so that the domain question can be asked with architecture held
    fixed: densenet_imagenet against densenet_cxr isolates pretraining.
    """

    in_channels = 3

    def __init__(self, layers=None, **_):
        super().__init__()
        from torchvision import models
        self.model = models.densenet121(weights="IMAGENET1K_V1")
        self.names = layers or ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
        self.freeze()

    def forward(self, x):
        h = self.prepare(x)
        f = self.model.features
        h = f.pool0(f.relu0(f.norm0(f.conv0(h))))
        out = []
        for i in (1, 2, 3, 4):
            h = getattr(f, f"denseblock{i}")(h)
            if f"denseblock{i}" in self.names:
                out.append(h)
            if i < 4:
                h = getattr(f, f"transition{i}")(h)
        return out




# --------------------------------------------------------------------------- #
# Loading a locally fine-tuned backbone
# --------------------------------------------------------------------------- #

# Wrapper keys a training harness might nest a state dict under. Tried in
# order; if none match, a state dict is identified by the shape of its keys,
# so an unfamiliar harness still loads.
WRAPPER_KEYS = ("features_state", "model_state", "state_dict", "model", "net",
                "features", "backbone")

COMMON_PREFIXES = ("module.", "model.", "backbone.", "vgg.", "net.", "encoder.")

_PARAM_SUFFIXES = (".weight", ".bias", ".running_mean", ".running_var",
                   ".num_batches_tracked")


def looks_like_state_dict(obj) -> bool:
    """True if the mapping's keys look like parameter names.

    Requires a majority of keys to end in a parameter suffix, so a config
    dict or a metrics dict sitting in the same checkpoint is not mistaken
    for weights.
    """
    if not isinstance(obj, dict) or not obj:
        return False
    keys = [k for k in obj if isinstance(k, str)]
    if len(keys) != len(obj):
        return False
    return sum(1 for k in keys if k.endswith(_PARAM_SUFFIXES)) >= max(1, len(keys) // 2)


def extract_state_dict(ckpt, return_key: bool = False):
    if isinstance(ckpt, dict):
        if looks_like_state_dict(ckpt):
            return (ckpt, None) if return_key else ckpt
        for key in WRAPPER_KEYS:
            inner = ckpt.get(key)
            if isinstance(inner, dict) and inner:
                return (inner, key) if return_key else inner
        for key, value in ckpt.items():
            if looks_like_state_dict(value):
                return (value, key) if return_key else value
    return (ckpt, None) if return_key else ckpt


def strip_common_prefix(sd: dict):
    stripped, changed = [], True
    while changed and sd:
        changed = False
        for pre in COMMON_PREFIXES:
            if all(k.startswith(pre) for k in sd):
                sd = {k[len(pre):]: v for k, v in sd.items()}
                stripped.append(pre)
                changed = True
                break
    return sd, stripped


def select_feature_keys(sd: dict) -> dict:
    feat = {k[len("features."):]: v for k, v in sd.items() if k.startswith("features.")}
    if feat:
        return feat
    return {k: v for k, v in sd.items() if k.split(".", 1)[0].isdigit()}


def describe_state_dict(sd: dict, limit: int = 10) -> str:
    keys = list(sd)
    lines = [f"{len(keys)} keys"]
    for k in keys[:limit]:
        v = sd[k]
        lines.append(f"    {k}  {tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")
    if len(keys) > limit:
        lines.append(f"    ... and {len(keys) - limit} more")
    return "\n".join(lines)


class VGG19CXR(FeatureExtractor):
    """VGG-19 trunk with locally fine-tuned chest X-ray weights.

    Two settings must come from the fine-tuning run rather than from defaults.

    INPUT CHANNELS is recovered automatically: it is read off the first
    convolution's weight shape, so a backbone fine-tuned on single-channel
    input skips the grayscale-to-RGB repeat.

    NORMALIZATION STATISTICS cannot be recovered from the weights. If the
    fine-tuning pipeline used anything other than ImageNet mean and standard
    deviation, leaving `normalization` unset silently shifts every feature
    map and this variant measures something other than what it claims. It
    does not raise. Set loss.perceptual.normalization in the config.

    To inspect a checkpoint before wiring it in:
        python perceptual.py --inspect /path/to/best_features.pt
    """

    input_size = None      # fully convolutional

    def __init__(self, checkpoint: str, layers=None, normalization=None,
                 strict: bool = True, **_):
        super().__init__()
        from pathlib import Path as _P
        from torchvision import models

        self.names = layers or ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]
        unknown = set(self.names) - set(VGG19_LAYERS)
        if unknown:
            raise ValueError(f"Unknown VGG-19 layer(s) {sorted(unknown)}")
        self.idx = [VGG19_LAYERS[n] for n in self.names]
        if self.idx != sorted(self.idx):
            raise ValueError(f"Layers must be in increasing depth order: {self.names}")
        max_idx = max(self.idx)

        path = _P(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Fine-tuned VGG-19 checkpoint not found: {path}\n"
                f"  Set loss.perceptual.checkpoint in the config.")

        raw = torch.load(path, map_location="cpu", weights_only=False)
        state, wrapper = extract_state_dict(raw, return_key=True)
        state, stripped = strip_common_prefix(state)
        feat = select_feature_keys(state)
        if not feat:
            raise RuntimeError(
                f"No convolutional trunk in {path} (wrapper: {wrapper}, "
                f"stripped: {stripped}).\n{describe_state_dict(state)}")

        first = feat.get("0.weight")
        if first is None or first.dim() != 4:
            raise RuntimeError(f"No '0.weight' in {path}.\n{describe_state_dict(feat)}")
        self.in_channels = int(first.shape[1])

        if normalization is not None:
            self.normalization = (tuple(normalization[0]), tuple(normalization[1]))
        elif self.in_channels == 1:
            raise ValueError(
                "This checkpoint expects 1-channel input, so 3-channel ImageNet "
                "statistics cannot apply. Set loss.perceptual.normalization to the "
                "mean/std the fine-tuning run used, e.g. [[0.5], [0.25]].")

        skeleton = models.vgg19(weights=None)
        if self.in_channels != 3:
            skeleton.features[0] = nn.Conv2d(self.in_channels, 64, 3, padding=1)
        self.features = nn.Sequential(*list(skeleton.features[: max_idx + 1]))

        keep = {k: v for k, v in feat.items()
                if k.split(".", 1)[0].isdigit() and int(k.split(".", 1)[0]) <= max_idx}
        res = self.features.load_state_dict(keep, strict=False)

        self.load_report = {
            "checkpoint": str(path), "wrapper_key": wrapper,
            "stripped_prefixes": stripped, "in_channels": self.in_channels,
            "loaded": len(keep), "missing": list(res.missing_keys),
            "normalization_from_config": normalization is not None,
        }
        if res.missing_keys and strict:
            raise RuntimeError(
                f"{len(res.missing_keys)} layers got no weights from {path}: "
                f"{res.missing_keys[:8]}. They would stay randomly initialised, "
                f"which makes the perceptual loss meaningless.")

        self.freeze()

    def forward(self, x):
        h, want, out = self.prepare(x), set(self.idx), []
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in want:
                out.append(h)
        return out



# --------------------------------------------------------------------------- #
# ResNet-50, ImageNet
# --------------------------------------------------------------------------- #

class ResNet50ImageNet(FeatureExtractor):
    """torchvision ResNet-50, ImageNet weights.

    Hooked after each residual stage, giving a four-scale hierarchy directly
    comparable to the VGG-19 and DenseNet selections. Paired with resnet_cxr
    it isolates pretraining domain with architecture held fixed, which is the
    only way to attribute a difference to the domain rather than to the
    network computing the features.
    """

    normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    in_channels = 3
    input_size = None          # fully convolutional

    def __init__(self, layers=None, weights: str = "IMAGENET1K_V2", **_):
        super().__init__()
        from torchvision import models
        stages = ["layer1", "layer2", "layer3", "layer4"]
        self.names = layers or list(stages)
        unknown = set(self.names) - set(stages)
        if unknown:
            raise ValueError(f"Unknown ResNet stage(s) {sorted(unknown)}. Available: {stages}")
        self.model = models.resnet50(weights=weights)
        self.freeze()

    def forward(self, x):
        h = self.prepare(x)
        m = self.model
        h = m.maxpool(m.relu(m.bn1(m.conv1(h))))
        out = []
        for name in ("layer1", "layer2", "layer3", "layer4"):
            h = getattr(m, name)(h)
            if name in self.names:
                out.append(h)
        return out


# --------------------------------------------------------------------------- #
# RAD-DINO (vision transformer)
# --------------------------------------------------------------------------- #

class RadDino(FeatureExtractor):
    """microsoft/rad-dino: ViT-B/14 trained on chest radiographs with DINOv2.

    THREE WAYS THIS DIFFERS FROM EVERY CONVOLUTIONAL BACKBONE HERE, each
    needing explicit handling rather than assumption.

      1. IT RETURNS TOKENS, NOT FEATURE MAPS. Each transformer block emits
         (B, 1 + N, D): one class token then N patch tokens. The class token
         is dropped and the patch tokens reshaped to (B, D, g, g) with g the
         grid side, so everything downstream (the perceptual loss and the
         feature-space residual alike) receives an ordinary feature map.

      2. THE INPUT SIZE MUST BE A MULTIPLE OF THE PATCH SIZE, 14. A 128 by 128
         image therefore cannot be fed directly. RAD-DINO was trained at 518,
         a 37 by 37 grid and 1,369 tokens. 224 is the standard DINOv2 fallback
         at 16 by 16 and 256 tokens, roughly a fifth of the cost, and DINOv2
         interpolates its positional embeddings so both are valid. 224 is the
         default here on cost grounds; set input_size: 518 for a fidelity
         check against the training resolution.

      3. LAYER SELECTION MEANS BLOCKS, NOT SCALES. ViT-B has twelve blocks all
         at one spatial resolution, so unlike a convolutional network the
         selection spans depth rather than scale. Blocks 3, 6, 9 and 12 by
         default. A perceptual loss over such features is therefore measuring
         something structurally different from the multi-scale VGG version,
         and the two are not directly interchangeable.

    Normalization is read from the image processor shipped with the weights
    rather than hardcoded, because getting it wrong shifts every feature map
    and raises nothing.

    SELF-SUPERVISED, NOT DISCRIMINATIVE. Unlike a classification fine-tune,
    DINOv2 training has no diagnostic decision boundary baked into it, so its
    features describe chest anatomy rather than separating diagnostic
    categories. That is the property a reconstruction loss wants.
    """

    in_channels = 3            # dinov2-base expects three channels
    input_range = "unit"

    def __init__(self, layers=None, weights: str = "microsoft/rad-dino",
                 input_size: int = 224, **_):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        if input_size % 14 != 0:
            raise ValueError(
                f"RAD-DINO has a patch size of 14, so input_size must be a "
                f"multiple of 14. Got {input_size}. Use 224 (16x16 tokens) or "
                f"518 (37x37, the training resolution).")
        self.input_size = input_size
        self.grid = input_size // 14

        self.model = AutoModel.from_pretrained(weights)
        proc = AutoImageProcessor.from_pretrained(weights)
        self.normalization = (tuple(proc.image_mean), tuple(proc.image_std))

        n = self.model.config.num_hidden_layers
        self.names = [int(i) for i in (layers or [n // 4, n // 2, 3 * n // 4, n])]
        bad = [i for i in self.names if not 1 <= i <= n]
        if bad:
            raise ValueError(f"RAD-DINO block indices must lie in 1..{n}; got {bad}")
        self.freeze()

    def forward(self, x):
        h = self.prepare(x)
        out = self.model(pixel_values=h, output_hidden_states=True)
        feats = []
        for k in self.names:
            # hidden_states[0] is the embedding output, so block k is index k.
            t = out.hidden_states[k][:, 1:, :]           # drop the class token
            b, n_tok, d = t.shape
            g = int(round(n_tok ** 0.5))
            if g * g != n_tok:
                raise RuntimeError(
                    f"RAD-DINO returned {n_tok} patch tokens, not a square grid. "
                    f"Check that input_size is a multiple of 14.")
            feats.append(t.transpose(1, 2).reshape(b, d, g, g))
        return feats

    def extra_repr(self) -> str:
        return (f"blocks={self.names}, input_size={self.input_size}, "
                f"grid={self.grid}x{self.grid}")



# torchvision ResNet child order, used to rename a checkpoint that was saved
# as an nn.Sequential. Such a file carries keys like "0.weight" and
# "4.0.conv1.weight" rather than "conv1.weight" and "layer1.0.conv1.weight",
# because nn.Sequential names its children by position. The shapes confirm the
# mapping: index 0 holds a (64, 3, 7, 7) kernel, which is conv1, and index 4
# holds bottleneck blocks, which is layer1.
RESNET_SEQUENTIAL_MAP = {
    "0": "conv1", "1": "bn1", "4": "layer1",
    "5": "layer2", "6": "layer3", "7": "layer4",
}


def remap_sequential_resnet(sd: dict) -> tuple[dict, bool]:
    """Renames nn.Sequential-style ResNet keys to torchvision names.

    Returns (state_dict, was_remapped). Keys whose leading component is not a
    recognised index are dropped, which removes the pooling and classifier
    entries a trunk does not need.
    """
    if not any(k.split(".", 1)[0].isdigit() for k in sd):
        return sd, False
    out = {}
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        name = RESNET_SEQUENTIAL_MAP.get(head)
        if name is None:
            continue                       # relu, maxpool, avgpool, fc
        out[f"{name}.{rest}" if rest else name] = v
    return out, True


class ResNet50RadImageNet(FeatureExtractor):
    """ResNet-50 pretrained on RadImageNet, via the Lab-Rasool PyTorch port.

    THE MIDDLE RUNG OF THE DOMAIN LADDER. Paired with resnet50_imagenet and
    resnet_cxr it gives three ResNet-50s that are architecturally identical
    and differ only in what they were pretrained on:

        resnet50_imagenet      natural photographs
        resnet50_radimagenet   medical imaging, but CT, MRI and ultrasound
        resnet_cxr             chest radiographs

    That is a controlled domain gradient, which a pair of backbones cannot
    provide.

    TWO CAVEATS WORTH RECORDING RATHER THAN DISCOVERING LATER.

    RadImageNet CONTAINS NO RADIOGRAPHS. Its 1.35 million images are CT, MRI
    and ultrasound across eleven anatomic regions. Those modalities differ
    from a chest film in image formation: CT is a reconstructed cross-section,
    ultrasound is speckle, a radiograph is a projection with superimposed
    structures. On low-level statistics a chest film is not obviously closer
    to a CT slice than to a photograph, so "medical" should not be read as
    "near-domain". The RadImageNet authors' own chest-radiography result is a
    1.9 percent AUC gain over ImageNet, the weakest transfer among the tasks
    they report, with the large gains concentrated in the modalities the
    dataset actually contains.

    IT IS A SUPERVISED CLASSIFIER. Like a diagnostic fine-tune, its features
    are shaped to separate pathology categories rather than to describe one
    category faithfully, which is the concern that applies to any
    discriminatively trained backbone used as a reconstruction loss. RAD-DINO
    is the self-supervised contrast to this.

    The checkpoint is an unofficial mirror rather than the original release,
    and is published as a pickled object rather than a plain state dict, so
    loading here accepts either form and reports what it found.
    """

    normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    in_channels = 3
    input_size = 224          # the published preprocessing is Resize(256), CenterCrop(224)

    def __init__(self, layers=None, repo_id: str = "Lab-Rasool/RadImageNet",
                 filename: str = "ResNet50.pt", checkpoint: str | None = None, **_):
        super().__init__()
        import torch.nn as _nn
        from torchvision import models

        stages = ["layer1", "layer2", "layer3", "layer4"]
        self.names = layers or list(stages)
        unknown = set(self.names) - set(stages)
        if unknown:
            raise ValueError(f"Unknown ResNet stage(s) {sorted(unknown)}. Available: {stages}")

        # A local path wins over the hub, so a cached copy can be used offline.
        if checkpoint:
            path = str(Path(checkpoint).expanduser())
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo_id, filename=filename)

        obj = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(obj, _nn.Module):
            # A pickled model object: use its trunk directly.
            self.model = obj
            self.load_report = {"source": path, "form": "nn.Module",
                                "normalization_from_config": True}
        else:
            # A state dict in some wrapper: unwrap, strip prefixes, load.
            sd, wrapper = extract_state_dict(obj, return_key=True)
            sd, stripped = strip_common_prefix(sd)
            # This release is saved as an nn.Sequential, so its keys are child
            # indices rather than torchvision layer names.
            sd, remapped = remap_sequential_resnet(sd)
            self.model = models.resnet50(weights=None)
            result = self.model.load_state_dict(sd, strict=False)
            missing = [k for k in result.missing_keys if not k.startswith("fc.")]
            self.load_report = {"source": path, "form": "state_dict",
                                "wrapper_key": wrapper, "stripped_prefixes": stripped,
                                "sequential_remap": remapped,
                                "missing": missing[:10],
                                "unexpected": list(result.unexpected_keys)[:10],
                                "normalization_from_config": True}
            if missing:
                raise RuntimeError(
                    f"{len(missing)} trunk parameters got no weights from {path}: "
                    f"{missing[:8]}. Those layers would stay randomly initialised, "
                    f"which makes the perceptual loss meaningless. Inspect the file "
                    f"with:\n{describe_state_dict(sd)}")

        if not hasattr(self.model, "layer4"):
            raise RuntimeError(
                f"The object loaded from {path} does not look like a torchvision "
                f"ResNet: no layer4 attribute. Found: "
                f"{list(dict(self.model.named_children()))[:10]}")
        self.freeze()

    def forward(self, x):
        h = self.prepare(x)
        m = self.model
        h = m.maxpool(m.relu(m.bn1(m.conv1(h))))
        out = []
        for name in ("layer1", "layer2", "layer3", "layer4"):
            h = getattr(m, name)(h)
            if name in self.names:
                out.append(h)
        return out

    def extra_repr(self) -> str:
        return f"stages={self.names}, input_size={self.input_size}"


REGISTRY = {
    "vgg19_imagenet": VGG19ImageNet,
    "vgg19_cxr": VGG19CXR,
    "densenet_imagenet": DenseNetImageNet,
    "densenet_cxr": DenseNetCXR,
    "resnet_cxr": ResNetCXR,
    "resnet50_imagenet": ResNet50ImageNet,
    "resnet50_radimagenet": ResNet50RadImageNet,
    "rad_dino": RadDino,
}


def build_extractor(name: str, **kwargs) -> FeatureExtractor:
    if name not in REGISTRY:
        raise ValueError(f"Unknown backbone {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


# --------------------------------------------------------------------------- #

class PerceptualLoss(nn.Module):
    """Feature-space distance. Not LPIPS: no channel-wise unit normalization
    and no learned per-channel weights. Keeping the two distinct matters
    because a calibrated LPIPS variant would be a separate experiment.

    Layers are normalised by their own magnitude before averaging. Without
    that, shallow layers with far larger feature maps dominate regardless of
    the weights, and the layer weights do not mean what they appear to.
    """

    def __init__(self, extractor: FeatureExtractor, layer_weights=None,
                 distance: str = "l2", normalize_layers: bool = True):
        super().__init__()
        self.extractor = extractor
        self.extractor.freeze()
        self.layer_weights = layer_weights
        self.distance = distance
        self.normalize_layers = normalize_layers

    def forward(self, x_hat, x):
        f_hat = self.extractor(x_hat)
        with torch.no_grad():
            f_x = self.extractor(x)

        w = self.layer_weights or [1.0] * len(f_hat)
        if len(w) != len(f_hat):
            raise ValueError(f"{len(w)} layer weights for {len(f_hat)} feature maps")

        total = x_hat.new_zeros(())
        for wi, a, b in zip(w, f_hat, f_x):
            d = self._distance(a, b)
            if self.normalize_layers:
                d = d / (b.detach().pow(2).mean() + 1e-8)
            total = total + wi * d
        return total / sum(w)

    def _distance(self, a, b):
        """One layer's distance between reconstruction and target features.

        WHY THE CHOICE MATTERS MORE THAN IT LOOKS

        l2 and l1 both require the reconstruction to put the right feature at
        the right POSITION. Averaged over a dataset, the value that minimises
        either is a per-position average of plausible features, which is the
        same conditional-mean argument that makes pixel MSE blurry, just moved
        one layer up. That is why a perceptual loss built on l2 improves LPIPS
        without measurably sharpening anything.

        gram is the one that breaks the pattern. It compares feature
        CORRELATIONS summed over all positions, so it asks for the right
        texture without saying where. A decoder can satisfy it by drawing
        rib-like structure anywhere, which is the mechanism by which style
        losses produce sharp output.

        THE COST OF THAT, stated plainly: a pure gram loss does not constrain
        placement at all, and an anomaly detector reads a spatially aligned
        residual. Use gram as an ADDITION to a positional term, not a
        replacement. l2+gram does exactly that.
        """
        if self.distance == "l2":
            return F.mse_loss(a, b)
        if self.distance == "l1":
            return F.l1_loss(a, b)
        if self.distance == "huber":
            # Quadratic near zero, linear beyond, so a few extreme feature
            # mismatches cannot dominate the layer.
            return F.smooth_l1_loss(a, b, beta=1.0)
        if self.distance == "cosine":
            # Direction only, per spatial position, so overall activation
            # magnitude is free. Useful when a backbone's scale varies across
            # the image and that variation is not what should be matched.
            an = F.normalize(a, dim=1, eps=1e-8)
            bn = F.normalize(b, dim=1, eps=1e-8)
            return (1.0 - (an * bn).sum(dim=1)).mean()
        if self.distance == "gram":
            return F.mse_loss(self._gram(a), self._gram(b))
        if self.distance == "l2+gram":
            # Positional term keeps the reconstruction aligned; the texture
            # term lets it be sharp. Equal halves, which is a starting point
            # rather than a tuned value.
            return 0.5 * F.mse_loss(a, b) + 0.5 * F.mse_loss(self._gram(a), self._gram(b))
        raise ValueError(f"distance must be l2, l1, huber, cosine, gram or "
                         f"l2+gram; got {self.distance!r}")

    @staticmethod
    def _gram(f):
        """Channel correlation matrix, normalised by the number of positions.

        Collapses H and W entirely, so what survives is which features occur
        together, not where. Dividing by (C*H*W) keeps the magnitude
        comparable across layers of very different sizes.
        """
        B, C, H, W = f.shape
        v = f.reshape(B, C, H * W)
        return torch.bmm(v, v.transpose(1, 2)) / (C * H * W)

# --------------------------------------------------------------------------- #

def _inspect(path: str) -> None:
    """Report what is inside a fine-tuned checkpoint, so it can be wired up
    without a failed training job in between."""
    from pathlib import Path as _P
    p = _P(path).expanduser()
    raw = torch.load(p, map_location="cpu", weights_only=False)
    print(f"File: {p}")
    if isinstance(raw, dict):
        print(f"Top-level keys: {list(raw)}")

    state, wrapper = extract_state_dict(raw, return_key=True)
    state, stripped = strip_common_prefix(state)
    print(f"Weights under wrapper key: {wrapper or '(top level)'}")
    print(f"Stripped prefixes: {stripped or 'none'}")
    feat = select_feature_keys(state)
    print(f"\nConvolutional trunk:\n{describe_state_dict(feat)}")

    first = feat.get("0.weight")
    if first is not None:
        c = first.shape[1]
        print(f"\nDetected in_channels = {c}")
        if c == 1:
            print("  -> 1-channel: loss.perceptual.normalization is REQUIRED "
                  "(single-element lists). ImageNet statistics cannot apply.")
        else:
            print("  -> 3-channel: grayscale input will be repeated across channels.")

    cfg = raw.get("config") if isinstance(raw, dict) else None
    if isinstance(cfg, dict):
        print("\n--- config embedded in the checkpoint ---")
        hints = ("normalization", "normalize", "norm", "mean", "std",
                 "image_mean", "image_std", "transform", "preprocessing")

        def walk(d, ind="  ", depth=0):
            if depth > 3:
                return
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f"{ind}{k}:")
                    walk(v, ind + "  ", depth + 1)
                else:
                    flag = "  <-- normalization?" if str(k).lower() in hints else ""
                    print(f"{ind}{k}: {v}{flag}")
        walk(cfg)
        print("\nCopy any mean/std above into loss.perceptual.normalization "
              "as [[mean...], [std...]].")
    else:
        print("\nNo config embedded. Normalization must come from the "
              "vgg_cxr_finetune source.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a fine-tuned backbone checkpoint")
    ap.add_argument("--inspect", type=str, required=True)
    args = ap.parse_args()
    _inspect(args.inspect)