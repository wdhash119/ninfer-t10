# Tesla T10: upstream merge of 2026-11-27 (269 commits), measured on the line

Host: `wsdev` (4× Tesla T10, TU102GL, `sm_75`, 56 SM, 16 GiB, PCIe Gen3, no NVLink), CUDA 13.4,
build `-DCMAKE_CUDA_ARCHITECTURES=75`, Release, tree `/public/src/ninfer-t10`.

* **Control**: the treatment column of [the 2026-11-20 merge report](tesla-t10-upstream-merge-2026-11-20.md)
  (tree `c6cde8f4`, build `2026-11-20.1-5-gc6cde8f4`), measured on this machine with these same launch
  flags and fixtures. That binary is not kept on the line (`/public/llm/bin` holds only
  `ninfer-serve-997b7cda`), so the control was not rerun; see [Not measured](#not-measured).
* **Treatment**: `8defe44a` — this tree after merging 269 upstream commits (upstream
  `2026-11-20.1` → `cbb6b024`, untagged), build `2026-11-20.1-276-g8defe44a`.

## What conflicted

Only `README.md`, and it keeps the fork's README as in `997b7cda` and `c6cde8f4`. The four files both
trees touched — `src/models/qwen3_5/execution/parameters.cpp`, `src/ops/linear/gguf/ggml_bridge.h`,
`src/ops/linear/gguf/gguf_linear.cpp`, `src/ops/linear/gguf/sources.cmake` — merged without hand
editing:

* `parameters.cpp` takes upstream's `GGUF_PTQ1_0` into the ternary token-table test beside the
  fork's `T2_G128_FP16`; the fork's rotation check stays.
* Upstream rewrote the GGUF vector path around it — `vector_columns()` (8 columns, 20 for PTQ1_0 on
  `sm_80`+), `vector_activation_transposed()`, and a batched `vector_products()` that takes a whole
  pass in one launch (`gguf_linear.cpp:341-370`, returning before the matrix loop). The fork's
  opt-in SM75 Marlin branch is still called where it was (`gguf_linear.cpp:250`, `:291`,
  `:406-419`; declarations at `ggml_bridge.h:153-157`). The two routes cannot preempt each other:
  Marlin asks for `columns >= 128` and only on `cc == 750`
  (`ggml_marlin_sm75.cu:174-185`), the batched vector path takes `t <= vector_columns(products)`,
  at most 20 (`ggml_bridge.h:60-66`).
* The merge deletes and renames nothing (`git diff --name-status 3a518735 HEAD` has no `D`/`R`
  rows), so no SM75 kernel was dropped; `ggml_marlin_sm75.cu`, `nvfp4_turing_qpn.cu` and the
  `w4_cublas` prefill units come through unchanged. 449 files, +56,983/−7,561.

## What the batch brings to a Turing card — and what it does not

* **The asynchronous MMQ tiles still cannot run here.** The batch adds `gguf_matrix/nvfp4`,
  `gguf_moe_matrix/nvfp4` and `gguf_matrix/ptq1_0` async keys, but inside the same
  `if (!volta && !turing_shapes)` block (`src/calibration/route_catalog.cpp:44-53`), so a `75`
  build proposes no new async route.
* **Ternary Bonsai 2.5 (`PTQ1_0`, `PQ2_0`) is dense-only and its fast forms are Ampere or newer.**
  Its MMQ tile configurations are registered only in `third_party/ggml-quants/src/ggml-cuda/
  mmq-config-ampere.cuh:36-43`, and its 20-column INT8 tensor-core vector form is guarded by
  `device_facts().cc >= 800` (`ggml_bridge_ptq1_tc.cuh:329`). A `75` build reads PTQ1_0 with the
  transposed lane kernel of at most 8 columns. Nothing here was measured on a Bonsai 2.5 artifact.
* **Inkling 2 Mini, Qwen4-exp, MXFP8 expert banks and the WDDM graph work** are Windows/Blackwell
  or model-family additions; this line runs Qwen3.8-27B on Linux and took none of them as a route
  change.
* **One thing did change under Turing: the route catalog.** The batch adds
  `gguf_moe_columns/<format>/{spread,whole}` for the 20 MoE vector block types *outside* the
  Volta/Turing guard (`route_catalog.cpp:38-42`), so the `sm_75` catalog digest moved from
  `b70799ec7029a5c2` to `04dc2a532ffaf48d`. See the next section.

## The catalog moved: the line had to be recalibrated

The calibrated `nvidia-tesla-t10-sm75` entry on this machine had been measured on 2026-11-13
against the old catalog, and an external entry whose `calibration.route_catalog` differs from the
build's digest is ignored (`src/runtime/engine/device_profile.cpp:287-292`). The first start of the
merged binary logged:

```
WARN engine | /home/devixtreme/.cache/ninfer/device-profiles.json: the entry for
  nvidia-tesla-t10-sm75 was calibrated against route catalog b70799ec7029a5c2
  (build 2026-11-12.4-50-gc0321db0), this build's is 04dc2a532ffaf48d; it is ignored.
  Recalibrate (ninfer-calibrate, or --device-profile calibrate)
INFO engine | calibrating routes for nvidia-tesla-t10-sm75 (56 SMs)
```

and took 273 s to `listening` instead of ~40 s, because the engine measured the routes itself on
every start. `build/apps/ninfer-calibrate --device 0` then measured **125 keys in 236.0 s, 104 of
them routed off the compiled route**, and stored them under the new digest:

```
stored 125 measured keys for nvidia-tesla-t10-sm75 (56 SMs, route catalog 04dc2a532ffaf48d)
  in /home/devixtreme/.cache/ninfer/device-profiles.json
```

The entry records `ninfer_build 2026-11-20.1-6-g3a518735-dirty` because `ninfer-calibrate` was run
from the merged working tree a minute before the merge commit was made; its catalog digest is the
one the committed build prints, so the entry is the one the measured binary loads below.

Against the 2026-11-13 entry (kept as
`profiles/bench/t10-upstream-merge-20261127/device-profiles.tesla-t10.bak-b70799ec-87-keys.json`,
alongside the new `…stored-04dc2a532ffaf48d.json`; `profiles/` is not tracked by git, these files
live on this machine):

* **38 keys added, all `gguf_moe_columns/*`, none dropped** — 87 routed keys become 125.
* **29 of the 87 shared keys changed bands.** 23 are `attn_i8_small/*`: the long-window bands
  `1048576` moved from `8x2x32p` to `8x2x32q` / `4x2x32qe` / `12x1x32qe`, i.e. the calibration no
  longer picks the pipelined `32p` variant the 2026-11-13 pass picked. The candidates in those
  bands sit within a few percent of each other, and two calibrations of *this same build* — the
  engine's own startup calibration and `ninfer-calibrate` eight minutes later — already disagreed
  on 11 keys (`attn_i8_small/h24/*`, `gdn_two_stage/h48`, `q5_linear_add/*`, `q6_head/*`,
  `q8_ksplit/*`, `t2_i8_small/*`), which bounds how much of the 29 is jitter. **Re-measure before
  trusting any single calibration of these keys.**

Startups after the recalibration log `device profile nvidia-tesla-t10-sm75: 125 routed keys
(ninfer-calibrate on Tesla T10)` four times, with no catalog note, in 39.0 s.

## Method

`profiles/bench/t10-upstream-merge-20261127/measure_merged.py` (the 2026-11-20 harness with one arm)
starts the merged binary with the production flags of `/public/llm/ninfer-qwen3.8-27b.sh` verbatim
(TP4 `1,1,1,1`, int8 prompt partials, `--prefill-cublas --prefill-chunk 4096`,
`--embedding-residency host`, `--kv-dtype int8 --kv-capacity auto`, `--max-context 262144`,
`--max-concurrency 2 --concurrent-prefill`, `--spec mtp --draft-tokens 3 --adaptive-mtp
--mtp-attention-window 4096`, `--vision`, `NINFER_GGUF_MARLIN=0`), only host/port/log paths differ.
Two passes, greedy, `seed 20261127`, one server per pass; each request carries a distinct
"Measurement pass …" marker line so the prefix cache never answers a measured request, and every
point recorded `prefix_cache_hit_tokens 0`. Rates are the server's own
`computed_prefill_tokens / prefill_s` and `(completion_tokens − 1) / decode_s` from
`--request-log-jsonl`. Mean ± σ of the two passes.

## Result

| Point | Prefill tok/s (2026-11-20 → this tree) | Decode tok/s (2026-11-20 → this tree) |
|---|---|---|
| 7,693-token retrieval, 17 tokens out | 1186.1 ± 2.2 → **1207.8 ± 7.9** (+1.83 %) | 122.73 ± 0.23 → **130.41 ± 0.03** (+6.26 %) |
| 64,525-token retrieval, 17 tokens out | 1070.3 ± 2.5 → **1074.6 ± 2.7** (+0.40 %) | 112.38 ± 0.18 → **120.79 ± 0.17** (+7.49 %) |
| 348-token prompt, 512 thinking tokens out (MTP3) | 804.0 ± 0.1 → **808.8 ± 4.9** (+0.60 %) | 92.95 ± 1.90 → **99.65 ± 0.70** (+7.21 %) |

Supporting state: startup 39.0 s (2026-11-20: 41.0 s); peak device memory 12,031 MiB on device 0 and
10,361 MiB on devices 1–3; `engine ready | total 36.2s | weights 19.2 GiB`;
`capacity | KV 524,288 tokens, int8, auto | pages 8,192/8,192 | runtime 5.97 GiB | free 3.81 GiB`;
`device 0 | weights 5.78 GiB | state 4.65 GiB | workspace 826.5 MiB` — the same footprint and
capacity lines as the 2026-11-20 report. MTP3 acceptance on the 512-token point 62.8 % → 64.2 %
(175.5 → 174.5 rounds).

Greedy answers are identical between the two trees on both retrieval points
(`ORCHID=493817; COLOR=COBALT`, `finish_reason stop_token`, 17 tokens) and identical between the
two passes of this tree; the 512-token thinking answer diverges from its third word as it did
between passes of the control, which is a greedy decode sitting on a near-tie.

**The prefill numbers say the merge costs nothing on this line: all three deltas are inside the
pass-to-pass spread the same binary shows.** The +6…7 % on decode does *not* belong to the merged
code alone: the 2026-11-20 numbers ran on the 2026-11-13 87-key calibration, while these run on a
fresh 125-key calibration whose `attn_i8_small/*` bands pick different INT8 small-T attention
variants. A strict attribution needs the pre-merge binary rebuilt as the control; what is
established here is that the merged tree plus its recalibration is at least as fast as the previous
pair, with the same outputs.

## Tests

`ctest --test-dir build -j4 -R 'ninfer_device_profile_test|ninfer_serve_options_test|
ninfer_request_log_test|ninfer_route_coverage_test|ninfer_kv_capacity_test|
ninfer_launch_config_test|ninfer_cli_options_test|ninfer_attn_input_proj_legacy_routes_test|
ninfer_attn_input_small_t_legacy_routes_test|ninfer_gdn_input_proj_legacy_routes_test|
ninfer_moe_route_test|ninfer_w4_cublas_prefill_test'` → **12/12 passed**
(`profiles/bench/t10-upstream-merge-20261127/tests-targeted.log`), the same set the 2026-11-20
report used.

A second, wider set of 17 GGUF/linear/attention cases (`tests-ops-broad.log`) passes 15/17. The two
failures are `75`-build refusals that predate the merge, not regressions:
`ninfer_linear_nvfp4_int8_test` and `ninfer_linear_swiglu_nvfp4_test` end in
`this build selects the legacy Linear table only`, thrown by
`src/ops/linear/common/route_table.cpp:238` — which already read that way at `3a518735` — and the
second one skips every case with `NVFP4 A4 execution requires an sm_120a GPU`.

The whole build (1,383 targets, arch `75`) compiled with **0 errors** and 764 `#`-class nvcc
warnings, 720 of them `#445-D` ("template parameter … is not used") and 44 `#177-D` ("variable
was declared but never referenced"), the same categories the 2026-11-20 report lists.

## Build prerequisite the batch adds

`cmake/NinferFFmpeg.cmake` now refuses to configure unless the pinned media archives are present,
and the batch added Mbed TLS 4.3.0 to them (`cmake/MediaSources.cmake`) for the WebUI's `https://`
MCP relay. On a checkout that only had FFmpeg and dav1d, configure fails with
`third_party/media-sources/mbedtls-4.3.0.tar.bz2 is missing`; `scripts/fetch-media-sources.sh`
downloads and digest-checks it (7.2 MiB, needs network access to github.com once). The other two
archives were already pinned at 9.0.2 and 1.5.4 and were not re-fetched.

## Not measured

* **No control rerun.** The `c6cde8f4` binary is not kept, so the comparison is against recorded
  numbers from the same machine, flags and fixtures rather than an interleaved A/B; the deltas
  therefore carry any host drift between 2026-11-20 and 2026-11-27 (nothing visible: the prefill
  points moved ≤ 1.8 % and the capacity lines are identical).
* Concurrent requests (`--max-concurrency 2` is set but the points are sequential), the vision path,
  MoE and every family but Qwen3.8-27B, prefix-cache hits, and answer quality beyond these prompts.
* Bonsai 2.5 / PTQ1_0, Inkling 2 Mini, MXFP8 experts, Qwen3.8-Flash-Next and the qwen4_exp Turing
  prompt attention (#438), all new upstream code paths that this line does not exercise.
* Architectures other than `75`; upstream's Ampere/Blackwell paths are compiled out here, not
  merely unused.

## Operations

* The stored device profile was replaced by the recalibrated 125-key entry
  (`route_catalog 04dc2a532ffaf48d`, build `2026-11-20.1-6-g3a518735-dirty`). The pre-merge
  87-key entry is kept as
  `profiles/bench/t10-upstream-merge-20261127/device-profiles.tesla-t10.bak-b70799ec-87-keys.json`;
  it is inert for this build (catalog mismatch) but still drives the pre-merge binaries.
* Rollback: point `NINFER_BIN` in `/public/llm/ninfer-qwen3.8-27b.sh` at
  `/public/llm/bin/ninfer-serve-997b7cda` (two merges back, still working). A pre-merge binary with
  the old catalog will simply ignore the new stored entry and calibrate at startup.
* No start failures in this run: both passes started in 39.0 s, and the shutdown after each pass was
  clean (`server stopped`, all four devices back to 0 MiB).
