# Optional CPU/FPGA interval profiling

> SUPERSEDED 2026-09-19: the initial recorder was incomplete (barrier double
> counting and no token-window selection). Its source was archived to
> /home/gyq/workspace_gyq/experiments/timeline_20260919/superseded_20260918.
> All initial instrumentation was removed from this final-source tree.
> Use /home/gyq/workspace_gyq/experiments/timeline_20260919/README_zh.md
> and its component-specific diagnostic sources. The instructions below are
> historical and must not be used for measurements.

This instrumentation is disabled by default.  With the default environment,
the runtime does not allocate a trace buffer, take a profiling timestamp for
each operator, or acquire the trace mutex.  It therefore does not belong in a
normal performance run.

## Enable a diagnostic trace

Set these variables only for a dedicated attribution run on the KV260:

```bash
export GGML_TIMELINE_TRACE=1
export GGML_TIMELINE_TRACE_PATH=/tmp/ggml_prefill_or_decode.csv
export GGML_TIMELINE_TRACE_MAX_EVENTS=1000000
```

The trace is flushed when the process exits.  The output is CSV and contains
absolute intervals normalized to the first recorded event:

```text
event_id,kind,phase,tid,start_us,end_us,duration_us,m,n,k,op,category,label
```

Check the interval arithmetic on the board or on the copied trace with:

```bash
python3 tools/analyze_ggml_timeline.py /tmp/ggml_prefill_or_decode.csv
```

## Measurement meaning

The records are nested, not additive across all rows.  The parent rows define
the wall intervals and the leaf rows provide an additive breakdown:

```text
CPU_GRAPH
  CPU_OP per worker thread
  CPU_BARRIER per worker thread

FPGA_GRAPH
  FPGA_NODE
    host_prepare
    activation_quant
    acc_submit_execute_writeback
    output_dequant
```

`acc_submit_execute_writeback` is deliberately named as one interval in this
first implementation: it is the exact wall interval of the TPU runtime call,
including host submission, accelerator execution and output completion.  It
does not fabricate a split that the runtime does not yet expose as interval
events.  The existing TPU aggregate counters remain the source for its
internal sub-counters; a later board-side pass can add nested runtime events
for descriptor programming, wait-for-done and output synchronization.

For the requested closed ledger, use the following boundaries:

- prefill: one `CPU_GRAPH`/`FPGA_GRAPH` interval for the selected representative
  graph, with CPU worker intervals and FPGA leaf stages nested inside it;
- decode: a continuous interval beginning at the selected token-512 graph and
  ending after the selected long decode window, not a sum of isolated GEMM
  calls;
- CPU and FPGA leaf rows may be summed only within the same parent graph and
  only after checking overlap on the parallel timeline.

The trace is diagnostic.  Report its attribution separately from the clean
throughput run, and compare `GGML_TIMELINE_TRACE=0` versus `1` if the trace
overhead itself needs to be quantified.
