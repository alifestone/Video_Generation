import sys, random, json, struct, numpy as np, gguf
from gguf.quants import dequantize
out, ref, src = sys.argv[1:4]
a, b = gguf.GGUFReader(out), gguf.GGUFReader(ref)
ta = [(t.name, t.tensor_type.name, t.shape.tolist()) for t in a.tensors]
tb = [(t.name, t.tensor_type.name, t.shape.tolist()) for t in b.tensors]
print("count", len(ta), len(tb), "identical name/type/shape:", ta == tb)
for k in ("general.architecture", "general.file_type", "general.quantization_version"):
    fa, fb = a.get_field(k), b.get_field(k)
    print(k, fa.parts[fa.data[-1]].tolist() if fa.types[0] != gguf.GGUFValueType.STRING else bytes(fa.parts[fa.data[-1]]).decode(), "ref:", fb.parts[fb.data[-1]].tolist() if fb.types[0] != gguf.GGUFValueType.STRING else bytes(fb.parts[fb.data[-1]]).decode())
fh = open(src, "rb"); n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n
big = [t for t in a.tensors if t.tensor_type.name == "Q8_0"]
random.seed(1)
for t in random.sample(big, 5) + [x for x in a.tensors if x.tensor_type.name == "F16" and len(x.shape) == 2][:1] + [a.tensors[-1]]:
    h = hdr[t.name]; s, e = h["data_offsets"]; fh.seek(base + s)
    ref32 = np.frombuffer(fh.read(e - s), dtype=np.float32).reshape(h["shape"])
    d = dequantize(t.data, t.tensor_type) if t.tensor_type.name == "Q8_0" else np.asarray(t.data)
    d = np.asarray(d, dtype=np.float32).reshape(ref32.shape)
    print(t.name, t.tensor_type.name, "rel_fro_err=%.5f" % (np.linalg.norm(d - ref32) / np.linalg.norm(ref32)), "max_abs=%.5f" % np.abs(d - ref32).max())
