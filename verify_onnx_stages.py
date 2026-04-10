import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn

from sp.configs import Config
from sp.configs import Dataset
from sp.configs import Initialization
from sp.configs import Model
from sp.configs import Size
from sp.configs import TransformerConfig
from sp.data.dvs_gesture.datamodule import DVSGestureDataModule
from sp.data_types import TokensBatch
from sp.dvs_gesture import NUM_CLASSES
from sp.nn.transformer_classifier import TransformerClassifier

CKPT_DIR = "experiments/DG_nano/checkpoints"
CKPT_PATH = f"{CKPT_DIR}/best.ckpt"

config = Config(
    dataset=Dataset.dvsgesture,
    buckets=10,
    batch_size=1,
    model=Model.transformer,
    patch_size=16,
    transformer=TransformerConfig(size=Size.nano, init=Initialization.random),
)
classifier = TransformerClassifier(config=config, num_classes=NUM_CLASSES)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state_dict = {
    k[len("model."):]: v
    for k, v in ckpt["state_dict"].items()
    if k.startswith("model.") and isinstance(v, torch.Tensor)
}
incompatible = classifier.load_state_dict(state_dict)
if incompatible.missing_keys:
    print(f"WARNING: missing keys: {incompatible.missing_keys}")
if incompatible.unexpected_keys:
    print(f"WARNING: unexpected keys: {incompatible.unexpected_keys}")
classifier.eval()

# Load a real sample from the test set
datamodule = DVSGestureDataModule(config, augment=False)
datamodule.setup("test")
# Select the 5th sample
loader = iter(datamodule.test_dataloader())
for _ in range(0):
    next(loader)
tb: TokensBatch = next(loader).inputs


tokens = tb.tokens[:1].float()
pos_x = tb.pos_x[:1].float()
pos_y = tb.pos_y[:1].float()
pos_t = tb.pos_t[:1].float()
padding_mask = tb.padding_mask[:1]
BATCH = tokens.shape[0]


def export_and_compare(
    name: str,
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    dynamic_axes: dict,
    output_names: list[str] | None = None,
) -> torch.Tensor:
    path = f"{CKPT_DIR}/stage_{name}.onnx"

    torch.onnx.export(
        module,
        inputs,
        path,
        input_names=input_names,
        output_names=output_names or ["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(path))

    with torch.no_grad():
        torch_out = module(*inputs).numpy()

    session = ort.InferenceSession(path)
    onnx_out = session.run(
        None,
        {k: v.numpy() for k, v in zip(input_names, inputs)},
    )[0]

    diff = np.abs(torch_out - onnx_out).max()
    print(f"[{name}] shape: {torch_out.shape}  max abs diff: {diff:.6f}")

    return torch.tensor(torch_out)


# ── Stage 1: Positional Encoding ─────────────────────────────────────────────
pos_out = export_and_compare(
    name="positional_encoding",
    module=classifier.positional,
    inputs=(pos_x, pos_y, pos_t),
    input_names=["pos_x", "pos_y", "pos_t"],
    dynamic_axes={
        "pos_x": {0: "batch", 1: "seq"},
        "pos_y": {0: "batch", 1: "seq"},
        "pos_t": {0: "batch", 1: "seq"},
    },
)

# ── Stage 2: Token Embedding ──────────────────────────────────────────────────
tok_out = export_and_compare(
    name="token_embedding",
    module=classifier.embeddings,
    inputs=(tokens,),
    input_names=["tokens"],
    dynamic_axes={"tokens": {0: "batch", 1: "seq"}},
)

# ── Stage 3: Encoder (4 heads + final LayerNorm on CLS) ──────────────────────
# Build the combined embedding the encoder actually receives
with torch.no_grad():
    combined = tok_out + pos_out
    cls_token = classifier.cls.detach().expand(BATCH, -1, -1)
    enc_input = torch.cat([cls_token, combined], dim=1)
    enc_padding = torch.cat(
        [torch.zeros(BATCH, 1, dtype=torch.bool), padding_mask],
        dim=1,
    )


class EncoderOnly(torch.nn.Module):
    """Encoder without LayerNorm — outputs the raw CLS token state."""

    def __init__(self, model: TransformerClassifier):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(embeddings, padding_mask)
        return encoded[:, 0]


enc_out = export_and_compare(
    name="encoder",
    module=EncoderOnly(classifier).eval(),
    inputs=(enc_input, enc_padding),
    input_names=["embeddings", "padding_mask"],
    dynamic_axes={
        "embeddings":   {0: "batch", 1: "seq"},
        "padding_mask": {0: "batch", 1: "seq"},
    },
    output_names=["cls_state"],
)

# ── Stage 4: Classification Head ──────────────────────────────────────────────
export_and_compare(
    name="classification_head",
    module=classifier.head,
    inputs=(enc_out,),
    input_names=["cls_state"],
    dynamic_axes={"cls_state": {0: "batch"}},
    output_names=["logits"],
)
