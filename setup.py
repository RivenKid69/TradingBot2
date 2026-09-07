# setup.py
"""
AI-Powered Quantitative Research Platform - Native Extensions Build System

Builds Cython/C++ extensions for high-performance trading simulation.
Supports Windows/Linux/macOS with reproducible builds.

Usage:
    python setup.py build_ext --inplace  # Build in-place for development
    python setup.py build_ext             # Build in build/ directory
    make build                            # Use Makefile (recommended)

Requirements:
    - Python 3.12+
    - Cython 3.0.10
    - numpy 1.26.4
    - C++17 compatible compiler (MSVC 19.x, GCC 11+, Clang 14+)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from Cython.Build import cythonize
except ImportError as e:
    raise SystemExit(
        "Cython is required to build extensions.\n"
        "Install it first: pip install -r requirements-build.txt"
    ) from e

# ============================================================================
# Build Configuration
# ============================================================================

# Version pinning for reproducibility
REQUIRED_CYTHON_VERSION = "3.0.10"
REQUIRED_NUMPY_VERSION = "1.26.4"
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# Compiler version tracking
COMPILER_INFO = {
    "python_version": PYTHON_VERSION,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
}


def get_compiler_version() -> str:
    """Detect and return compiler version for reproducibility."""
    if sys.platform.startswith("win"):
        # Try to detect MSVC version
        try:
            import setuptools.msvc

            vc_env = setuptools.msvc.msvc14_get_vc_env(platform.machine())
            return vc_env.get("VCToolsVersion", "unknown")
        except Exception:
            return "MSVC-unknown"
    else:
        # Try to get gcc/clang version
        import subprocess

        try:
            result = subprocess.run(
                ["gcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            first_line = result.stdout.split("\n")[0]
            return first_line
        except Exception:
            try:
                result = subprocess.run(
                    ["clang", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                first_line = result.stdout.split("\n")[0]
                return first_line
            except Exception:
                return "unknown"


COMPILER_INFO["compiler_version"] = get_compiler_version()


# ============================================================================
# Custom Build Command with Numpy and Hash Reporting
# ============================================================================


class BuildExtWithNumpyAndHash(build_ext):
    """Custom build_ext that:
    1. Adds numpy include directories at build time
    2. Generates build hash report for reproducibility
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Defer numpy import until build time (so it's available in venv)
        import numpy as _np

        numpy_version = _np.__version__
        if numpy_version != REQUIRED_NUMPY_VERSION:
            print(f"WARNING: numpy {numpy_version} != {REQUIRED_NUMPY_VERSION} (pinned)")

        self.include_dirs = (self.include_dirs or []) + [_np.get_include()]

    def run(self):
        """Run build and generate hash report."""
        # Check Cython version
        from Cython import __version__ as cython_version

        if cython_version != REQUIRED_CYTHON_VERSION:
            print(f"WARNING: Cython {cython_version} != {REQUIRED_CYTHON_VERSION} (pinned)")

        # Run standard build
        build_ext.run(self)

        # Generate hash report
        self.generate_hash_report()

    def generate_hash_report(self):
        """Generate SHA256 hash report of all built extensions."""
        report = {
            "build_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "python_version": PYTHON_VERSION,
                "cython_version": REQUIRED_CYTHON_VERSION,
                "platform": COMPILER_INFO["platform"],
                "machine": COMPILER_INFO["machine"],
                "compiler": COMPILER_INFO["compiler_version"],
            },
            "extensions": {},
        }

        # Hash all built shared libraries
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            if os.path.exists(ext_path):
                sha256_hash = self._compute_file_hash(ext_path)
                report["extensions"][ext.name] = {
                    "path": ext_path,
                    "sha256": sha256_hash,
                    "size_bytes": os.path.getsize(ext_path),
                }

        # Write report
        report_path = Path("build_hash_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Build Hash Report: {report_path.absolute()}")
        print(f"{'='*70}")
        print(f"Built {len(report['extensions'])} extensions:")
        for name, info in report["extensions"].items():
            print(f"  {name:30s} SHA256: {info['sha256'][:16]}...")
        print(f"{'='*70}\n")

    @staticmethod
    def _compute_file_hash(filepath: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# ============================================================================
# Platform-Specific Compiler Flags
# ============================================================================

if sys.platform.startswith("win"):
    # MSVC flags for Windows
    c_args = [
        "/O2",  # Maximum optimization
        "/W3",  # Warning level 3
        "/GL",  # Whole program optimization
    ]
    cxx_args = [
        "/O2",
        "/std:c++17",
        "/W3",
        "/GL",
        "/EHsc",  # Exception handling
    ]
    link_args = ["/LTCG"]  # Link-time code generation
elif sys.platform.startswith("darwin"):
    # macOS (clang)
    c_args = [
        "-O3",
        "-fvisibility=hidden",
        "-march=native",  # Optimize for this CPU
    ]
    cxx_args = [
        "-O3",
        "-std=c++17",
        "-fvisibility=hidden",
        "-march=native",
    ]
    link_args = ["-std=c++17"]
else:
    # Linux (gcc/clang)
    c_args = [
        "-O3",
        "-fvisibility=hidden",
        "-march=native",
        "-mtune=native",
    ]
    cxx_args = [
        "-O3",
        "-std=gnu++17",  # gnu++17 for <algorithm>, <random>, etc.
        "-fvisibility=hidden",
        "-march=native",
        "-mtune=native",
    ]
    link_args = ["-std=gnu++17"]

include_dirs = ["."]
library_dirs = []
libraries = []


# ============================================================================
# Extension Modules
# ============================================================================

# C++ source files (shared across multiple extensions)
orderbook_cpp = "OrderBook.cpp"
market_sim_cpp = "MarketSimulator.cpp"
micro_cpp = "cpp_microstructure_generator.cpp"

ext_modules = [
    # ====== Pure Cython Extensions (C backend) ======
    Extension(
        name="obs_builder",
        sources=["obs_builder.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="info_builder",
        sources=["info_builder.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
    ),
    Extension(
        name="environment",
        sources=["environment.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
    ),
    Extension(
        name="coreworkspace",
        sources=["coreworkspace.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="execevents",
        sources=["execevents.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="execaction_interpreter",
        sources=["execaction_interpreter.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="execlob_book",
        sources=["execlob_book.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="execengine",
        sources=["execengine.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="execfast_execution",
        sources=["execfast_execution.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="risk_manager",
        sources=["risk_manager.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
    ),
    Extension(
        name="micromicrogen",
        sources=["micromicrogen.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    # ====== Cython + C++ Extensions ======
    Extension(
        name="fast_lob",
        sources=["fast_lob.pyx", orderbook_cpp],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="fast_market",
        sources=["fast_market.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="reward",
        sources=["reward.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="micro_sim",
        sources=["micro_sim.pyx", orderbook_cpp, micro_cpp],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="marketmarket_simulator_wrapper",
        sources=["marketmarket_simulator_wrapper.pyx", market_sim_cpp],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="lob_state_cython",
        sources=[
            "lob_state_cython.pyx",
            orderbook_cpp,
            market_sim_cpp,
            micro_cpp,
        ],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
]


# ============================================================================
# Setup Configuration
# ============================================================================

setup(
    name="ai-quant-platform-extensions",
    version="1.0.0",
    description="Cython/C++ extensions for AI-Powered Quantitative Research Platform",
    author="Sultan Suyunbayev",
    author_email="sulastomatolog@gmail.com",
    license="Apache-2.0",
    python_requires=">=3.12",
    py_modules=["apply_no_trade_mask", "no_trade"],
    entry_points={
        "console_scripts": ["no-trade-mask=apply_no_trade_mask:main"],
    },
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
            "annotation_typing": True,
        },
        annotate=False,  # Set to True to generate HTML annotation files
    ),
    cmdclass={"build_ext": BuildExtWithNumpyAndHash},
)
