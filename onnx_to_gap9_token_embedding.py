import glob
import subprocess
import numpy as np
import onnx

CKPT_DIR = "experiments/DG_nano/checkpoints"

# Get Inputs
embeddings   = np.load(f"{CKPT_DIR}/stage_encoder_input_embeddings.npy")
padding_mask = np.load(f"{CKPT_DIR}/stage_encoder_input_padding_mask.npy")

print(f"embeddings:   {embeddings.shape} {embeddings.dtype}")
print(f"padding_mask: {padding_mask.shape} {padding_mask.dtype}")

# Get Outputs
output = np.load(f"{CKPT_DIR}/stage_encoder_output_onnx.npy")

print(f"output: {output.shape} {output.dtype}")

output_gap9 = np.fromfile("tmp/gap9_token_embedding/Output_1.bin", dtype=np.float16).reshape(output.shape)
print(f"output_gap9: {output_gap9.shape} {output_gap9.dtype}")
def snr(a,b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = np.abs(a - b)
    print(f"Max diff: {diff.max()}")
    print("SNR:", 10 * np.log10(np.sum(a**2) / np.sum(diff**2)))
snr(output, output_gap9)
exit()

# Get ONNX
model_path = f"{CKPT_DIR}/stage_encoder.onnx"
model = onnx.load(model_path)
graph = model.graph

# NNTools

from nntool.api import NNGraph
from nntool.api.utils import quantization_options

G = NNGraph.load_graph(model_path)
G.adjust_order()
G.fusions('float_match_group')
G.fusions('expression_matcher')
G.quantize(None, graph_options=quantization_options(scheme='float', float_type='float16'))
G.draw(filepath="quantized", view=False)

def print_model_infos(model):
    max_activ_size, total_params = model.total_memory_usage
    ops = model.total_ops
    print(f"\tMax Active Size:\t{max_activ_size} elements")
    print(f"\tTotal # Parameters:\t{total_params} elements")
    print(f"\tTotal # Operations:\t{ops / 1e6:.2f} MOps")
print_model_infos(G)

fout = G.execute((embeddings, padding_mask))
#qout = G.execute((embeddings, padding_mask), dequantize=True, quantize=True)

# Set allocate to 1 to treat the variable as scratch memory

for input_node in G.input_nodes():
    print(input_node.name)
    input_node.allocate = 1
    #input_node.at_options.out_home_mem_loc = "AT_MEM_L3_DEFAULTRAM"

for output_node in G.output_nodes():
    output_node.allocate = 1

# Step 1: Generate code + tensors only (dont_run=True skips cmake build and GVSOC).
# We need this split because NNTool has a bug: for CNN_BatchedMatMulAct_fp16 with
# KOP_MATMUL, AutoTiler allocates H2 items for the bias, but NNTool serializes only
# W2 items (head_dim=32) to the tensor file. The kernel expects seq_len=66 items.
print('Generating GAP9 code + tensors...')
G.execute_on_target(
    pmsis_os='freertos',
    platform='gvsoc',
    directory="tmp/gap9_token_embedding",
    input_tensors=[embeddings, padding_mask],
    write_out_to_file=True,
    at_log=True,
    dont_run=True,
    settings={
        'l1_size':108000,
        'l2_size':1200000,
        'l3_size':8000000,
        #'l3_flash_mb': 8,
        #'l3_ram_device': "AT_MEM_L3_DEFAULTRAM",
        #'l3_flash_device': "AT_MEM_L3_DEFAULTFLASH",
        #'privileged_l3_flash_size': int(0.8*1024*1024),
        #'privileged_l3_flash_device': "AT_MEM_L3_MRAMFLASH",
        #'l3_flash_mb':4,
        #'l3_flash_device':"AT_MEM_L3_DEFAULTFLASH",
        #"graph_l2_static_memory_budget" : 150000,
        'tensor_directory':'./tensors',
        'graph_const_exec_from_flash':True,
        'graph_size_opt': 2,
        #'graph_async_fork':True,
        #"graph_dump_tensor": 6,          # dump inputs + outputs of every node
        #"graph_dump_tensor_to_file": True, # write to files instead of printf
        #'graph_l2_static_memory_budget': 200000,
        #"graph_warm_construct" : 3
        'graph_checksum':True
    },
    do_clean=False,
    cmake=True,
    at_loglevel=2,
    print_output=True,
    #check_on_target=True,
    #verbose = True,
    #tolerance=0.001
)

# Step 2: Patch NNTool bug — matmul_2_biases written with 32 (head_dim) items but
# AutoTiler allocates H2=66 (seq_len) items for the bias of this kernel.
# The operation softmax@V has no seq-length bias in standard attention, so zeros are correct.
_EXPECTED_BYTES = 66 * 2  # 66 float16 items = 132 bytes
for _path in glob.glob("tmp/gap9_token_embedding/tensors/*self_attn_matmul_2_biases.tensor"):
    _data = open(_path, 'rb').read()
    if len(_data) != _EXPECTED_BYTES:
        with open(_path, 'wb') as _f:
            _f.write(bytes(_EXPECTED_BYTES))
        print(f"Patched {_path}: {len(_data)//2} → 66 float16 items (zero bias)")

# Step 3: Build and run GVSOC with patched tensors.
print('Executing on GAP9 Target - GVSOC...')
subprocess.run(
    ["bash", "run_make.sh"],
    cwd="tmp/gap9_token_embedding",
    check=True,
)

