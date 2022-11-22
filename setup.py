#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reddemcee",
    version="0.3.0",
    author="ReddTea",
    author_email="redd@tea.com",
    description="A Parallel Tempering wrapper for emcee 3 for personal use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    #data_files=[(path1, [path1+'/exo_list.csv', path1+'/ss_list.csv'])],
    include_package_data=True,
    install_requires=['numpy', 'matplotlib>=3.5.1', 'emcee', 'tqdm', 'pandas'],
    python_requires=">=3.6",
)
