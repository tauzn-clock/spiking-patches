import numpy as np
import onnx

CKPT_DIR = "experiments/DG_nano/checkpoints"

# Get Inputs
pos_x = np.load(f"{CKPT_DIR}/stage_positional_encoding_input_pos_x.npy")
pos_y = np.load(f"{CKPT_DIR}/stage_positional_encoding_input_pos_y.npy")
pos_t = np.load(f"{CKPT_DIR}/stage_positional_encoding_input_pos_t.npy")

# Get Outputs
output = np.load(f"{CKPT_DIR}/stage_positional_encoding_output_onnx.npy")

print(f"output: {output.shape} {output.dtype}")

# Get ONNX
model_path = f"{CKPT_DIR}/stage_positional_encoding.onnx"
model = onnx.load(model_path)
graph = model.graph

# Get NNTools

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

for input_node in G.input_nodes():
    print(input_node.name)
    input_node.allocate = 1
    #input_node.at_options.out_home_mem_loc = "AT_MEM_L3_DEFAULTRAM"

for output_node in G.output_nodes():
    print(output_node.name)
    output_node.allocate = 1
    #output_node.at_options.out_home_mem_loc = "AT_MEM_L3_DEFAULTRAM"

#Execute on Target (aka invoke AutoTiler)
print('Executing on GAP9 Target - GVSOC...')
res = G.execute_on_target(
    pmsis_os='freertos',
    platform='gvsoc',
    directory="tmp/gap9_pos_encoding",
    input_tensors=[pos_x, pos_y, pos_t],
    write_out_to_file=True,
    at_log=True,
    dont_run=False,
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

output_gap9 = np.fromfile("tmp/gap9_pos_encoding/Output_1.bin", dtype=np.float16).reshape(output.shape)
print(output_gap9.shape, output_gap9.dtype)
print(output.shape)

print(output_gap9[0, :10])
print(output[0, :10])
def snr(a,b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = np.abs(a - b)
    print(f"Max diff: {diff.max()}")
    print("SNR:", 10 * np.log10(np.sum(a**2) / np.sum(diff**2)))
snr(output, output_gap9)