"""Unified reward layer for SandboxFusion-RL.

Two RL scenarios, one import surface:

  Test-generation RL (model writes a pytest suite for a fixed reference):
      correctness, statement/branch coverage, per-test mutation score
      -> suite_rewards(...)   [async, needs a prebuilt mutant pool]

  Code-generation RL (model writes a solution for fixed I/O tests):
      pass/fail correctness + deterministic instruction-count efficiency
      -> solution_rewards(...) [sync, perf_event_open via the sandbox]

Both talk to the same SandboxFusion server (default http://localhost:8080,
override with SANDBOX_FUSION_ENDPOINT).
"""
from .compute import solution_rewards, suite_rewards  # noqa: F401
