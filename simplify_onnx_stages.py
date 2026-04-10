import glob

import onnx
import onnxsim

CKPT_DIR = "experiments/DG_nano/checkpoints"

# Main purpose is to remove MOD found in torch.nn.MultiheadAttention, which is not supported by Autotiller
for path in sorted(glob.glob(f"{CKPT_DIR}/stage_*.onnx")):
    model = onnx.load(path)
    model_simplified, check = onnxsim.simplify(model)
    assert check, f"Simplification failed for {path}"
    onnx.save(model_simplified, path)
    print(f"Simplified {path}")


