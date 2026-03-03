"""Verbose logging setup with -v, -vv, -vvv support."""

import argparse
import logging
import sys
from typing import Optional


def setup_logging(verbosity: int = 0) -> logging.Logger:
    """Configure logging based on verbosity level.

    Args:
        verbosity: 0 = WARNING, 1 = INFO, 2+ = DEBUG

    Returns:
        Configured logger instance
    """
    level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }.get(min(verbosity, 2), logging.DEBUG)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    return logging.getLogger(__name__)


def add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    """Add -v/-vv/-vvv argument to argument parser."""
    parser.add_argument(
        "-v",
        "-vv",
        "-vvv",
        action="count",
        default=0,
        help="Increase verbosity: -v=info, -vv=debug, -vvv=trace",
    )


def get_verbose_level(args: argparse.Namespace) -> int:
    """Get verbosity level from parsed arguments."""
    return getattr(args, "v", 0)
