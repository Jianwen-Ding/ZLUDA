# Independent public-header client: prepared, not yet verified

Current status: the reference client has NOT been compiled or run against a complete independent RTX
header set. Existing prefix probes and four offline tool tests do not establish that result. Direct HTTPS
access to NVIDIA documentation is refused in the current shell environment. The public NVIDIA/TensorRT-RTX
GitHub repository currently contains samples/demos, not the required interface header set.

Two tools prepare a reproducible check using public Apache-2.0 interface source only:

```sh
python3 zluda_tensorrt_agx/tools/snapshot_public_headers.py --output-dir /scratch/rtx-1.7.1-public
python3 zluda_tensorrt_agx/tools/check_reference_abi.py \
  --reference-dir /scratch/rtx-1.7.1-public \
  --manifest /scratch/rtx-1.7.1-public/manifest.json \
  --abi-library /path/to/libtensorrt_agx_runtime_abi.so \
  --native-library /path/to/libtensorrt_agx.so \
  --output-dir /scratch/reference-client
```

The snapshot tool extracts source lines from public Doxygen HTML, checks the Apache SPDX marker and each
page's RTX 1.7.1 label, follows NvInfer header dependencies, checks version macros, and records source and
HTML SHA-256 hashes plus URLs. The source URLs use latest, so they are mutable: the resulting hashes pin
that particular obtained snapshot, not a vendor-signed release archive. Mixed-version pages fail. Previously
obtained documentation HTML can be supplied with --html-dir. No shared library, SDK, vendor engine or
proprietary binary is fetched. This extraction has only synthetic/offline parser testing so far.

The compile/run tool requires the manifest and all recorded header hashes/licenses to match. It includes the
independent reference directory and copies only our unrelated native C declarations into the output directory;
it does not add our C++ subset include directory. External CUDA type declarations, if required, can be supplied
as a separately reviewed --type-include directory. This needs only interface types, not libcudart or a vendor
runtime. No proprietary dependencies are bundled or downloaded by these tools.

The client uses the actual reference wrapper classes and callbacks, dynamically resolves our runtime factory,
and exercises runtime configuration, custom-header validity, custom engine deserialization, I/O metadata,
runtime config, serialization config, host-memory ownership and names-only inspection. Its fixture is built
with our native C builder API. It does not exercise builder/network/layer binary interfaces, plugins, GPU context
execution or complete ABI coverage. --sanitize instruments the client only; use an instrumented adapter build
as well for adapter coverage. A successful result is a targeted runtime-subset check, never full completion.

Offline tool checks:

```sh
python3 zluda_tensorrt_agx/tests/test_public_header_tools.py
```

Four pass: source-line/HTML entity extraction, missing-source rejection, missing-license rejection and
missing-header fail-closed behavior. The actual public-header compile/run gate remains outstanding.
