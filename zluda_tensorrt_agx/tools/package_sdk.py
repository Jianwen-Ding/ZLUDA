#!/usr/bin/env python3
"""Package only our inference implementation, headers, example and portable kernel artifacts."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--target-dir', type=Path, required=True, help='Cargo release directory')
parser.add_argument('--kernels-dir', type=Path, required=True, help='Output of build_kernels.py --emit both')
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--cxx', default='c++')
args = parser.parse_args()
crate = Path(__file__).resolve().parents[1]
repo = crate.parent
out = args.output_dir.resolve()
if out.exists() and any(out.iterdir()):
    parser.error("output directory must be empty; use a fresh directory for each package")
# Fail before writing anything if any required input is missing.
files = {
    args.target_dir / 'libtensorrt_agx.so': 'lib/libtensorrt_agx.so',
    args.target_dir / 'ztrt-compile': 'bin/ztrt-compile',
    crate / 'examples/native.cpp': 'examples/native.cpp',
    crate / 'README.md': 'REFERENCE.md',
    repo / 'agx_libs/kernels/inference.mlir': 'kernels/inference.mlir',
}
for name in ['zluda_tensorrt.h', 'NvInfer.h', 'NvOnnxParser.h']:
    files[crate / 'include' / name] = 'include/' + name
for name in ['LICENSE-MIT', 'LICENSE-APACHE']:
    files[repo / name] = name
for name in ['inference.air', 'inference.air.ll', 'inference.agxfat']:
    files[args.kernels_dir / name] = 'kernels/' + name
for source in files:
    if not source.is_file(): parser.error(f'missing required input: {source}')
for source, relative in files.items():
    dest = out / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
subprocess.run([args.cxx, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                '-I', str(out / 'include'), str(out / 'examples/native.cpp'),
                '-L', str(out / 'lib'), '-Wl,-rpath,$ORIGIN/../lib',
                '-ltensorrt_agx', '-ldl', '-o', str(out / 'bin/ztrt-example')], check=True)
(out / 'README.md').write_text('''# Independent MLIR/AIR inference SDK

This package contains our original library and its source-level TensorRT API subset.
It uses its own ZTRTAGX1 engines. It does not load opaque NVIDIA engines and is not
binary-compatible with libtensorrt_rtx.so or libnvinfer.so. See REFERENCE.md for scope.

Run the included independently built MLP:

```sh
ZLUDA_AGX_LIBCUDA=/absolute/path/to/our/libcuda.so ./bin/ztrt-example
```

Expected output: [11, 11, 0, 0]. Our ZLUDA AGX CUDA driver must be available; no
NVIDIA runtime, CUDA toolkit or libcudart installation is needed.

Compile public ONNX without a GPU:

```sh
./bin/ztrt-compile --format onnx model.onnx model.ztrt
```

Compile your own application with `-I include -L lib -ltensorrt_agx` and set
an application-appropriate rpath or LD_LIBRARY_PATH for this SDK's lib directory.
The example has a relative rpath, so this package can be moved to another directory.
Portable MLIR, upstream AIR bitcode/text and AGX fat objects are in kernels/.
macOS AIR packaging/downgrading is a separate step.
''')
manifest = []
for path in sorted(out.rglob('*')):
    if path.is_file() and path.name != 'SHA256SUMS':
        manifest.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out)}\n')
(out / 'SHA256SUMS').write_text(''.join(manifest))
print(f'Packaged independent inference SDK: {out}')
