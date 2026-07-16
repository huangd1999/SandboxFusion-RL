# SandboxFusion (Improved Fork)

This is a fork of [bytedance/SandboxFusion](https://github.com/bytedance/SandboxFusion) with additional improvements and bug fixes.

## Installation

Please refer to the original repository for installation instructions: [SandboxFusion README](https://github.com/bytedance/SandboxFusion/blob/main/README.md)

## Improvements

### 1. Command Line Arguments Support

Added `extra_args` feature to pass command line arguments to the executed program. This allows not only passing input via `stdin`, but also passing arguments like `./test -arg1 val1`.

### 2. Sandbox Concurrency Control

Implemented actual concurrency control for sandbox execution. You can now configure `max_concurrency` in `local.yaml` to limit the number of concurrent sandbox instances.

### 3. Bug Fixes

- Fixed buffer overflow issue caused by large `stdin`/`stdout` data filling up the buffer.

## Contributing

Contributions are welcome! Feel free to submit pull requests for further improvements.