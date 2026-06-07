# Agent Entry Point

Read [CLAUDE.md](CLAUDE.md) first. It contains the workspace-level MLX fork
guide: multi-repo structure, TurboQuant goals, validation expectations,
benchmark rules, promotion gates, and handoff practices.

This repo is the C++/Metal MLX core fork. Changes here often need matching
wiring in `mlx-c`, `mlx-swift`, and `mlx-swift-lm` before downstream claims are
valid.
