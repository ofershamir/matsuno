#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:02:46 2019

@author: shlomi
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymaws",
    version="0.1.0",
    author="Ofer Shamir",
    author_email="ofer.shamir@mail.huji.ac.il",
    description="Matsuno Analytical Wave Solution implemented in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ofershamir/matsuno",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)