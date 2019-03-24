#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:02:46 2019

@author: shlomi
"""
from distutils.core import setup

setup(
    name='pymaws',
    packages=['pymaws'],
    version='0.1.0',  # Ideally should be same as your GitHub release tag varsion
    description='Matsuno Analytical Wave Solution implemented in Python',
    author='Ofer Shamir',
    author_email='ofer.shamir@mail.huji.ac.il',
    url='https://github.com/ofershamir/matsuno',
    download_url='https://github.com/ofershamir/matsuno/archive/v0.1.0.tar.gz',
    keywords=['matsuno', 'gravity-inertia-waves'],
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
)
