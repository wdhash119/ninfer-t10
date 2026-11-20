# Tesla T10: upstream merge of 2026-11-20 (45 commits), measured on the line

Host: `wsdev` (4× Tesla T10, TU102GL, `sm_75`, 56 SM, 16 GiB, PCIe Gen3, no NVLink), CUDA 13.4,
build `-DNINFER_CUDA_ARCHITECTURES=75`, Release, tree `/public/src/ninfer-t10`.

* **Control**: `997b7cda`, build `2026-11-13.2-24-g997b7cda`, binary
  `/public/llm/bin/ninfer-serve-997b7cda` (the binary the service ran before this merge).
* **Treatment**: `c6cde8f4` — this tree after merging 45 upstream commits (upstream tags
  `2026-11-19.1`, `2026-11-19.2`, `2026-11-20.1`), build `2026-11-20.1-5-gc6cde8f4`.

Only `README.md` conflicted. Upstream's rewrite of `gguf_project` (the trailing
`const void* quantized_x` argument, `Activation::data` becoming `const void*`, the pre-quantized
activation pushed first) merged into the fork's opt-in SM75 Marlin branch without hand editing, and
`marlin_sm75_supported` / `marlin_sm75_activation_bytes` / `marlin_sm75_activation` /
`marlin_sm75_product` are still called where they were (`src/ops/linear/gguf/gguf_linear.cpp:227`,
`:266`, `:347-357`; declarations at `src/ops/linear/gguf/ggml_bridge.h:121-125`). The merge deletes
and renames nothing (`git diff --name-status 997b7cda HEAD` has no `D`/`R` rows), so no SM75 kernel
was dropped. 170 files, +8574/−1194.

## What the batch brings to a Turing card — and what it does not

The batch's headline is the asynchronous MMQ tile loop (`third_party/ggml-quants/src/ggml-cuda/
mmq.cuh`, `cp-async.cuh`) with per-format route keys `gguf_matrix/<fmt>`, `gguf_moe_matrix/<fmt>`.
It cannot run here: `ggml_cuda_mmq_async_tiles_available()` requires compute capability ≥ 80
(`mmq.cuh:918`) and `MMQ_ASYNC_TILES_AVAILABLE` is guarded by `__CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE`,
and the route-catalog entries for those keys are added only under `if (!volta && !turing_shapes)`
(`src/calibration/route_catalog.cpp:33-40`) with `turing_shapes(Turing) == true`
(`src/ops/common/device_tier.h:107`). So a `75` build never proposes an async route, and — because
that block is the only change to the catalog — the catalog this build computes is the same as the
control's. That matters because the T10 runs on its own measured calibration, not on a built-in
profile: there is no `nvidia-tesla-t10-sm75` row in `src/runtime/engine/device_profiles.json`, and
an external entry whose recorded `calibration.route_catalog` differs from the build's digest is
ignored. Both binaries therefore load the same file
(`/home/devixtreme/.cache/ninfer/device-profiles.json`, digest `b70799ec7029a5c2`, calibrated
2026-11-13) and log, four times each, `engine | device profile nvidia-tesla-t10-sm75: 87 routed keys
(calibration)` with no catalog-mismatch or recalibration note. **No recalibration was needed.**
Upstream did re-measure the built-in `sm_75` profiles on 2026-11-19 (`nvidia-geforce-rtx-2080-ti-sm75`
at 68 SM gained `gguf_vector/q4_k`, `gguf_moe_*`, `attn_prompt_split` and `w4_cublas/*` bands); the
56-SM T10 does not match that hardware class, so those bands are not applied here.

Everything else that could move the needle is measured below: the small-T attention P-fragment
change (`src/ops/softmax_attention/dense/causal_cache/*/small_t.cuh`), the elementwise wrapper
refactors (`src/ops/wrapper/{rmsnorm,layer_norm,rope,gdn_gating,causal_conv1d_silu,add_bias}.cpp`),
the KV append work, and the FP16 `--kv-dtype fp16` option (this line stays on `int8`).

## Method

`profiles/bench/t10-upstream-merge-20261120/ab_prod.py` starts each binary with the production flags
of `/public/llm/ninfer-qwen3.8-27b.sh` verbatim (TP4 `1,1,1,1`, int8 prompt partials,
`--prefill-cublas --prefill-chunk 4096`, `--kv-dtype int8 --kv-capacity auto`, `--max-context
262144`, `--max-concurrency 2 --concurrent-prefill`, `--spec mtp --draft-tokens 3 --adaptive-mtp
--mtp-attention-window 4096`, `--vision`, `NINFER_GGUF_MARLIN=0`), only host/port/log paths differ.
Two passes, control then treatment inside each pass, greedy, `seed 20261120`, one server per pass.
Each request carries a distinct "Measurement pass …" marker line so the prefix cache never answers a
measured request; every point recorded `prefix_cache_hit_tokens 0` and the server's own
`prefill`/`decode` split from `--request-log-jsonl`. Rates are `computed_prefill_tokens / prefill_s`
and `(completion_tokens − 1) / decode_s`, i.e. engine-side, HTTP excluded. Mean ± σ of the two passes.

## Result

| Point | Prefill tok/s (control → treatment) | Decode tok/s (control → treatment) |
|---|---|---|
| 7,693-token retrieval, 17 tokens out | 1187.9 ± 15.2 → 1186.1 ± 2.2 (**−0.15 %**) | 122.81 ± 0.05 → 122.73 ± 0.23 (**−0.06 %**) |
| 64,525-token retrieval, 17 tokens out | 1073.1 ± 4.4 → 1070.3 ± 2.5 (**−0.26 %**) | 112.67 ± 0.24 → 112.38 ± 0.18 (**−0.26 %**) |
| 348-token prompt, 512 thinking tokens out (MTP3) | 791.0 ± 9.3 → 804.0 ± 0.1 (+1.64 %) | 94.03 ± 1.78 → 92.95 ± 1.90 (−1.14 %) |

Supporting state, identical on both sides: startup 41.0 s vs 39–43 s; peak device memory 12,029 MiB
per card; `capacity | KV 524,288 tokens, int8, auto | pages 8,192/8,192 | runtime 5.97 GiB | free
3.81 GiB`; `device 0 | weights 5.78 GiB | state 4.65 GiB | workspace 826.5 MiB`. MTP3 acceptance on
the 512-token point: 63.8 % → 62.8 % (335.5/526 drafted → 333.5/532), rounds 175.5 → 177.5.

The two retrieval points decide the question: both deltas are below the spread the *same* binary
shows between passes (control 8K prefill alone moved 1198.6 → 1177.1, 1.8 %). **The merge does not
cost throughput on this line.** The 512-token thinking point moves −1.1 % with σ 1.9 within one
binary (control 92.8/95.3, treatment 94.3/91.6), so its delta is inside its own noise; its acceptance
rate moved 1 pp, which is what a greedy decode sitting on a near-tie does.

Greedy output parity: the 8K and 64K answers are byte-identical in the recorded head for both passes.
The 512-token thinking answer differs from the third word onward — but the control pass 2 produced the
same text as the treatment pass 1, so the two builds sample both variants and no systematic change is
visible. Numerics on that point are not bit-identical, which is expected: the batch reorganises how
the small-T attention P fragments are loaded.

## Tests

`ctest --test-dir build -j4 -R 'ninfer_device_profile_test|ninfer_serve_options_test|
ninfer_request_log_test|ninfer_route_coverage_test|ninfer_kv_capacity_test|
ninfer_launch_config_test|ninfer_cli_options_test|ninfer_attn_input_proj_legacy_routes_test|
ninfer_attn_input_small_t_legacy_routes_test|ninfer_gdn_input_proj_legacy_routes_test|
ninfer_moe_route_test|ninfer_w4_cublas_prefill_test'` → 12/12 passed
(`profiles/bench/t10-upstream-merge-20261120/tests-targeted.log`). The full build (725 targets,
arch 75) compiled with 0 errors and 630 `#`-class warnings; the ones visible in the log are of
categories that predate the merge, e.g. `warning #445-D: template parameter "Geometry" is not used`
in `src/ops/softmax_attention/dense/causal_cache/geometry.cuh:67`. The control build's warning count
was not captured, so the count itself is not a comparison.

## Not measured

Concurrent requests (`--max-concurrency 2` was set but the points are sequential), the vision path,
MoE and non-Qwen families, prefix-cache hits (suppressed by design), answer quality beyond these
three prompts, and every architecture but `75` — this tree is built for the T10 only, so upstream's
Ampere/Blackwell async-MMQ paths are compiled out rather than merely unused here.

## Operations

The first start of the merged binary on the line failed once with
`ERROR cudaMalloc failed: cudaErrorMemoryAllocation: out of memory` five milliseconds into
`loading weights | 19.8 GiB`, with the process staying alive while holding 12.0 GiB on device 0 and
10.3 GiB on each other device; a second start attempt would then have found the memory occupied.
Killing that process (SIGTERM, gone in 3 s) and starting again succeeded on the first attempt with
the footprint and capacity lines above. Rollback: point `NINFER_BIN` in the launch script at
`/public/llm/bin/ninfer-serve-997b7cda`, which is kept as the control.
