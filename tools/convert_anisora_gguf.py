"""Convert Wan2.2 I2V A14B native F32 safetensors -> ComfyUI-GGUF Q8_0 GGUF.
Rules (tensor order/type) copied from a reference QuantStack GGUF. Streams tensor by tensor (low RAM).
Usage: python convert_anisora_gguf.py SRC.safetensors REF.gguf DST.gguf
"""
import sys, time, resource
import numpy as np, gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import quantize
import json, struct

class ST:
    """Minimal safetensors reader using read() (mmap of 57GB fails under overcommit limits)."""
    def __init__(self, path):
        self.fh = open(path, "rb")
        n = struct.unpack("<Q", self.fh.read(8))[0]
        self.hdr = json.loads(self.fh.read(n)); self.hdr.pop("__metadata__", None)
        self.base = 8 + n
    def keys(self): return self.hdr.keys()
    def shape(self, k): return tuple(self.hdr[k]["shape"])
    def get(self, k):
        h = self.hdr[k]; assert h["dtype"] == "F32", h["dtype"]
        a, b = h["data_offsets"]
        self.fh.seek(self.base + a)
        return np.frombuffer(self.fh.read(b - a), dtype=np.float32).reshape(h["shape"])

src, ref, dst = sys.argv[1:4]
r = gguf.GGUFReader(ref)
plan = [(t.name, t.tensor_type, tuple(int(x) for x in t.shape[::-1])) for t in r.tensors]

f = ST(src)
keys = set(f.keys())
assert keys == {p[0] for p in plan}, (keys ^ {p[0] for p in plan})

w = gguf.GGUFWriter(path=dst, arch="wan")
w.add_quantization_version(gguf.GGML_QUANT_VERSION)
w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)

for name, tt, shape in plan:
    assert f.shape(name) == shape, (name, shape)
    n = int(np.prod(shape))
    if tt == Q.Q8_0:
        nb = n // 32 * 34
        bshape = shape[:-1] + (shape[-1] // 32 * 34,)
        w.add_tensor_info(name, bshape, np.dtype(np.uint8), nb, raw_dtype=Q.Q8_0)
    elif tt == Q.F16:
        w.add_tensor_info(name, shape, np.dtype(np.float16), n * 2)
    elif tt == Q.F32:
        w.add_tensor_info(name, shape, np.dtype(np.float32), n * 4)
    else:
        raise ValueError(tt)

w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
t0 = time.time()
for i, (name, tt, shape) in enumerate(plan):
    a = f.get(name)
    if tt == Q.Q8_0:
        a = quantize(a, Q.Q8_0)
    elif tt == Q.F16:
        a = a.astype(np.float16)
    w.write_tensor_data(a)
    del a
    if i % 100 == 0:
        print(i, len(plan), f"{time.time()-t0:.0f}s maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f}GB", flush=True)
w.close()
print("done", f"{time.time()-t0:.0f}s maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f}GB")
