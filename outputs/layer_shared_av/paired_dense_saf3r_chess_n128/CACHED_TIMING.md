# SAF3R timing caveat

This initial paired run reset lazy DINO Top-K only before warmup. Consequently,
measured SAF3R forwards reused the warmup Top-K cache on identical inputs.
Its accuracy is valid for these same inputs, but the SAF3R latency omits that
per-forward selector work and MUST NOT be used as the complete SAF3R latency.
Dense timing has no such cache and is unaffected.

The benchmark runner now calls update_args_per_forward inside every timed
SAF3R iteration. Fresh, separately named baseline results are required.
