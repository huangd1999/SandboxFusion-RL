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

import logging
import os
import sys

import structlog

from sandbox.configs.run_config import RunConfig

config = RunConfig.get_instance_sync()

# Resolved once at import time. Override via SANDBOX_LOG_LEVEL=INFO/WARNING/...
_LOG_LEVEL = getattr(logging, os.environ.get('SANDBOX_LOG_LEVEL', 'DEBUG').upper(), logging.DEBUG)


def configure_logging(trace_file=None):

    def filter_keys(_, __, event_dict):
        event_dict.pop('_from_structlog', None)
        event_dict.pop('_record', None)
        return event_dict

    structlog.configure(processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
                        logger_factory=structlog.stdlib.LoggerFactory(),
                        context_class=dict)

    handlers = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(_LOG_LEVEL)
    stdout_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[filter_keys, structlog.dev.ConsoleRenderer(colors=config.common.logging_color)],))
    handlers.append(stdout_handler)

    if isinstance(trace_file, str):
        file_handler = logging.FileHandler(trace_file, 'w+')
        file_handler.setLevel(_LOG_LEVEL)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(processors=[filter_keys,
                                                            structlog.processors.JSONRenderer()],))
        handlers.append(file_handler)

    logging.basicConfig(level=_LOG_LEVEL, handlers=handlers)
    logging.getLogger('aiosqlite').setLevel(logging.CRITICAL)
    logging.getLogger('databases').setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
