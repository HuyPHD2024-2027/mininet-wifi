#!/usr/bin/env python3
"""
Setup script for Mininet-WiFi Client Component
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements_client.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mininet-wifi-client",
    version="1.0.0",
    author="Mininet-WiFi Team",
    author_email="ramon.fontes@ufrn.br",
    description="Client component for Mininet-WiFi network operations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/intrig-unicamp/mininet-wifi",
    packages=find_packages(include=["client*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Networking",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "mininet-wifi-client=client.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "client": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="mininet wifi networking sdn openflow client",
    project_urls={
        "Bug Reports": "https://github.com/intrig-unicamp/mininet-wifi/issues",
        "Source": "https://github.com/intrig-unicamp/mininet-wifi",
        "Documentation": "https://mininet-wifi.github.io/",
    },
)