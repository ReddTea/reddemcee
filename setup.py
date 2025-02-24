#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 26 aug 2024
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reddemcee",
    version="0.8.5",
    author="ReddTea",
    author_email="redd@tea.com",
    description="An Adaptative Parallel Tempering wrapper for emcee 3 for personal use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=['numpy', 'emcee>=3', 'tqdm'],
    python_requires=">=3.6",
)
