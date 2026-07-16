# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import os
import subprocess
import time
import traceback
from typing import Dict, List, Optional

import psutil
import structlog
import platform
import resource

from sandbox.configs.run_config import RunConfig
from sandbox.runners.isolation import tmp_cgroup, tmp_netns, tmp_overlayfs
from sandbox.runners.types import CodeRunArgs, CodeRunResult, CommandRunResult, CommandRunStatus
from sandbox.utils.common import set_permissions_recursively
from sandbox.utils.execution import cleanup_process, ensure_bash_integrity, get_output_non_blocking, kill_process_tree, try_decode

logger = structlog.stdlib.get_logger()
config = RunConfig.get_instance_sync()


def _force_kill_group(pid: int) -> None:
    """Best-effort kill of the entire process group rooted at ``pid``.

    With ``start_new_session=True`` the child becomes the session leader and
    process group leader (pgid == pid). We send SIGKILL to ``-pgid`` so every
    descendant — including the root-owned `python` inside `sudo bwrap` — dies
    immediately, even when the inner process is in a different PID namespace.

    Falls back to psutil-based traversal for the case where setsid was not
    applied (e.g. callers passing pre-existing PIDs).
    """
    import signal
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # PermissionError happens when bwrap children are root-owned.
        try:
            subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', f'-{pid}'],
                           timeout=5,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=False)
        except Exception as e:
            logger.warning(f'force-kill-pgrp failed for pgid={pid}: {e}')
    # Belt-and-braces: also do the psutil walk in case anything escaped the
    # process group (e.g. a daemonized child that called setsid itself).
    try:
        kill_process_tree(pid)
    except Exception:
        pass


async def run_command_bare(command: str | List[str],
                           timeout: float = 10,
                           stdin: Optional[str] = None,
                           cwd: Optional[str] = None,
                           extra_env: Optional[Dict[str, str]] = {},
                           use_exec: bool = False,
                           preexec_fn=None) -> CommandRunResult:
    try:
        logger.debug(f'running command {command}')
        # `start_new_session=True`: spawn the child as its own session leader
        # so it forms a new process group with pgid == child.pid. On timeout we
        # send SIGKILL to the whole group via `sudo kill -- -<pgid>` (sudo is
        # required because `sudo bwrap` makes the descendants root-owned and
        # the sandbox runs as an unprivileged user). Without this, LLM-emitted
        # infinite-loop code accumulates as un-killable root subprocesses,
        # blocking the agent loop and leaking memory.
        if use_exec:
            p = await asyncio.create_subprocess_exec(*command,
                                                     stdin=subprocess.PIPE,
                                                     stdout=subprocess.PIPE,
                                                     stderr=subprocess.PIPE,
                                                     env={
                                                         **os.environ,
                                                         **(extra_env or {})
                                                     },
                                                     preexec_fn=preexec_fn,
                                                     start_new_session=True)
        else:
            p = await asyncio.create_subprocess_shell(command,
                                                      stdin=subprocess.PIPE,
                                                      stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE,
                                                      cwd=cwd,
                                                      executable='/bin/bash',
                                                      env={
                                                          **os.environ,
                                                          **(extra_env or {})
                                                      },
                                                      preexec_fn=preexec_fn,
                                                      start_new_session=True)
        
        if stdin is not None:
            logger.debug(f"Passing stdin of length {len(stdin)} to communicate().")
            stdin_bytes = stdin.encode()
        else:
            stdin_bytes = None

        start_time = time.time()
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                p.communicate(input=stdin_bytes),
                timeout=timeout
            )
            execution_time = time.time() - start_time
            logger.debug(f'stop running command {command}')
        except asyncio.TimeoutError:
            # Timeout: terminate the process and read as much of the generated output as possible.
            try:
                # p.kill()
                _force_kill_group(p.pid)
            except ProcessLookupError:
                pass
            stdout_b, stderr_b = await p.communicate()

            return CommandRunResult(status=CommandRunStatus.TimeLimitExceeded,
                                    execution_time=time.time() - start_time,
                                    # stdout=await get_output_non_blocking(p.stdout),
                                    # stderr=await get_output_non_blocking(p.stderr))
                                    stdout=try_decode(stdout_b),
                                    stderr=try_decode(stderr_b))
        finally:
            if psutil.pid_exists(p.pid):
                _force_kill_group(p.pid)
                logger.info(f'process killed: {p.pid}')
            if config.sandbox.cleanup_process:
                cleanup_process()
            if config.sandbox.restore_bash:
                ensure_bash_integrity()

        return CommandRunResult(status=CommandRunStatus.Finished,
                                execution_time=execution_time,
                                return_code=p.returncode,
                                # stdout=await get_output_non_blocking(p.stdout),
                                # stderr=await get_output_non_blocking(p.stderr))
                                stdout=try_decode(stdout_b),
                                stderr=try_decode(stderr_b))
    except Exception as e:
        message = f'exception on running command {command}: {e} | {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        return CommandRunResult(status=CommandRunStatus.Error, stderr=message)


async def run_commands(compile_command: Optional[str], run_command: str, cwd: str, extra_env: Optional[Dict[str, str]],
                       args: CodeRunArgs, **kwargs) -> CodeRunResult:
    files = {}
    compile_res = None
    run_res = None

    if config.sandbox.isolation == 'none':
        preexec_steps = []
        if kwargs.get('set_uid'):
            set_permissions_recursively(cwd, 0o777)
            preexec_steps.append(lambda: os.setuid(kwargs.get('set_uid')))
        
        # Apply memory limit using resource module
        if args.memory_limit_MB > 0:
            def memory_limit_preexec():
                _, hard_memory_limit_AS = resource.getrlimit(resource.RLIMIT_AS)
                _, hard_memory_limit_DATA = resource.getrlimit(resource.RLIMIT_DATA)
                soft_memory_limit = args.memory_limit_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (soft_memory_limit, hard_memory_limit_AS))
                resource.setrlimit(resource.RLIMIT_DATA, (soft_memory_limit, hard_memory_limit_DATA))
                if platform.uname().system != "Darwin":
                    _, hard_memory_limit_STACK = resource.getrlimit(resource.RLIMIT_STACK)
                    resource.setrlimit(resource.RLIMIT_STACK, (soft_memory_limit, hard_memory_limit_STACK))
            preexec_steps.insert(0, memory_limit_preexec)
        preexec_fn = lambda: [step() for step in preexec_steps] if preexec_steps else None
        
        if compile_command is not None:
            compile_res = await run_command_bare(compile_command,
                                                 args.compile_timeout,
                                                 None,
                                                 cwd,
                                                 extra_env,
                                                 preexec_fn=preexec_fn)
        if compile_res is None or (compile_res.status == CommandRunStatus.Finished and compile_res.return_code == 0):
            run_res = await run_command_bare(run_command + " " + args.extra_args if args.extra_args else run_command,
                                             args.run_timeout,
                                             args.stdin,
                                             cwd,
                                             extra_env,
                                             preexec_fn=preexec_fn)
        for filename in args.fetch_files:
            fp = os.path.abspath(os.path.join(cwd, filename))
            if os.path.isfile(fp):
                with open(fp, 'rb') as f:
                    content = f.read()
                base64_content = base64.b64encode(content).decode('utf-8')
                files[filename] = base64_content
        return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)

    elif config.sandbox.isolation == 'lite':
        # Force netns_no_bridge=True: bridge MASQUERADE setup tries to call
        # iptables against the host's primary interface (auto-detected from
        # default route), which (a) requires elevated privilege beyond sudo on
        # some configurations and (b) reintroduces a path the sandbox code
        # could use to reach external services. Without the bridge the
        # subprocess has an isolated, deaf network — fits our threat model
        # (the sandbox should not need network for code-eval correctness).
        async with tmp_overlayfs() as root, tmp_cgroup(mem_limit='4G', cpu_limit=1) as cgroups, tmp_netns(
                True) as netns:
            prefix = []
            for cg in cgroups:
                prefix += ['cgexec', '-g', cg]
            # `unshare --pid` requires CAP_SYS_ADMIN; unprivileged user
            # namespaces are disabled on this kernel. Drive everything from
            # under a single sudo: chroot + unshare + ip netns are all root
            # operations once we accept they need elevation.
            #
            # `sudo -E` preserves the parent env (PATH especially) so the
            # conda runtime python from `extra_env['PATH']` is findable inside
            # the chroot — without -E sudo resets PATH to the secure_path,
            # which doesn't have /home/nvidia/miniconda3/envs/sandbox-runtime/bin.
            prefix += []
            if not kwargs.get('disable_pid_isolation', False):
                prefix += ['unshare', '--pid', '--fork', '--mount-proc']
            prefix += ['ip', 'netns', 'exec', netns]
            prefix += ['chroot', root]

            if compile_command is not None:
                compile_res = await run_command_bare(prefix + ['bash', '-c', f'cd {cwd} && {compile_command}'],
                                                     args.compile_timeout, None, cwd, extra_env, True)
            if compile_res is None or (compile_res.status == CommandRunStatus.Finished and
                                       compile_res.return_code == 0):
                run_res = await run_command_bare(prefix + ['bash', '-c', f'cd {cwd} && {run_command}'],
                                                 args.run_timeout, args.stdin, cwd, extra_env, True)

            for filename in args.fetch_files:
                fp = os.path.join(root, os.path.abspath(os.path.join(cwd, filename))[1:])
                if os.path.isfile(fp):
                    with open(fp, 'rb') as f:
                        content = f.read()
                    base64_content = base64.b64encode(content).decode('utf-8')
                    files[filename] = base64_content
            return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)

    elif config.sandbox.isolation == 'bwrap':
        # Bubblewrap-based isolation. On systems where the SandboxFusion
        # "lite" overlay+netns+cgroup chain is too tangled (cgroup v2,
        # kernel.unprivileged_userns_clone=0, non-standard NIC names, etc.),
        # bwrap gives us a single-binary, namespace-based sandbox that runs
        # the user code in a fresh mount/pid/net namespace with explicit
        # bind-mounts of only the system paths we trust.
        #
        # Requires sudo (since the kernel here disallows unprivileged user
        # namespaces); the local sudoers grants NOPASSWD: ALL.
        #
        # Threat model: defends against LLM-generated code that does
        # `rm -rf /data`, `os.unlink(...)`, etc. — those paths simply do
        # not exist inside the bwrap sandbox.
        bwrap_env_args = []
        merged_env = {**os.environ, **(extra_env or {})}
        # Explicitly set PATH so the sandbox-runtime python in
        # /home/nvidia/miniconda3/envs/sandbox-runtime/bin is findable.
        for k in ('PATH', 'LANG', 'LC_ALL', 'HOME'):
            if k in merged_env:
                bwrap_env_args += ['--setenv', k, merged_env[k]]
        bwrap_prefix = [
            'sudo', 'bwrap',
            '--ro-bind', '/usr', '/usr',
            '--ro-bind', '/lib', '/lib',
            '--ro-bind', '/lib64', '/lib64',
            '--ro-bind-try', '/bin', '/bin',
            '--ro-bind-try', '/sbin', '/sbin',
            # Bind only specific /etc files — NOT all of /etc, otherwise
            # /etc/shadow, /etc/sudoers, etc. become readable from inside
            # the sandbox. Python needs hosts/resolv.conf/passwd/group for
            # name lookups and pwd/grp module imports.
            '--ro-bind-try', '/etc/hosts', '/etc/hosts',
            '--ro-bind-try', '/etc/resolv.conf', '/etc/resolv.conf',
            '--ro-bind-try', '/etc/passwd', '/etc/passwd',
            '--ro-bind-try', '/etc/group', '/etc/group',
            '--ro-bind-try', '/etc/ssl', '/etc/ssl',
            '--ro-bind-try', '/etc/ca-certificates', '/etc/ca-certificates',
            '--ro-bind-try', '/etc/timezone', '/etc/timezone',
            '--ro-bind-try', '/etc/localtime', '/etc/localtime',
            '--ro-bind-try', '/etc/nsswitch.conf', '/etc/nsswitch.conf',
            # The sandbox-runtime conda env lives under /home/nvidia/miniconda3
            # and is the `python` we want to invoke.
            '--ro-bind', '/home/nvidia/miniconda3', '/home/nvidia/miniconda3',
            # Bind the user-code cwd writable so the test .py file and any
            # compile artifacts are visible/writable.
            '--bind', cwd, cwd,
            '--proc', '/proc',
            '--dev', '/dev',
            '--chdir', cwd,
            '--unshare-pid',
            '--unshare-net',
            '--die-with-parent',
            '--clearenv',
        ] + bwrap_env_args + ['--']

        # Per-execution memory cap. LLM-generated code sometimes writes
        # accidental megabombs (e.g., `arr = [0] * n*n` with n=10**5), which
        # at 64 concurrent reward workers can each grow to hundreds of GB RSS
        # — enough to OOM the 2 TB host and kill the trainer.
        # Budget math: usable RAM ≈ 1.6 TB (after training + OS), max
        # concurrent sandbox calls = REWARD_NUM_WORKERS = 64. Theoretical
        # ceiling 25 GB/call; we cap at 8 GB to leave ~1.1 TB margin for
        # OS/buffer/burst. 8 GB is ~16× headroom over typical CP solutions
        # (~500 MB worst-case dp/dict cases).
        # ulimit -v sets RLIMIT_AS on the bash subshell, inherited by the
        # user's python/cpp process. Honor the caller's memory_limit_MB when
        # provided, otherwise default to 8 GB.
        mem_limit_mb = args.memory_limit_MB if args.memory_limit_MB > 0 else 8192
        ulimit_prefix = f"ulimit -v {mem_limit_mb * 1024}; "  # -v is in KB

        if compile_command is not None:
            compile_res = await run_command_bare(
                bwrap_prefix + ['bash', '-c', ulimit_prefix + compile_command],
                args.compile_timeout, None, cwd, extra_env, True)
        if compile_res is None or (compile_res.status == CommandRunStatus.Finished and
                                   compile_res.return_code == 0):
            run_res = await run_command_bare(
                bwrap_prefix + ['bash', '-c',
                                ulimit_prefix + run_command + (" " + args.extra_args if args.extra_args else "")],
                args.run_timeout, args.stdin, cwd, extra_env, True)

        for filename in args.fetch_files:
            fp = os.path.abspath(os.path.join(cwd, filename))
            if os.path.isfile(fp):
                with open(fp, 'rb') as f:
                    content = f.read()
                base64_content = base64.b64encode(content).decode('utf-8')
                files[filename] = base64_content
        return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)


def restore_files(dir: str, files: Dict[str, Optional[str]]):
    for filename, content in files.items():
        if not isinstance(content, str):
            continue
        if "IGNORE_THIS_FILE" in filename:
            continue
        filepath = os.path.join(dir, filename)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as file:
            file.write(base64.b64decode(content))
