#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.org').read()

requirements = ['numpy', 'scipy']

setup(
    name='vad',
    version='0.1.0',
    description='Python port of Bowon Lee\'s Voice Activity Detector',
    long_description=readme + '\n\n',
    author='Maarten Versteegh',
    author_email='maartenversteegh@gmail.com',
    url='https://github.com/mwv/vad',
    packages=[
        'vad',
    ],
    package_dir={'vad':
                 'vad'},
    include_package_data=True,
    install_requires=requirements,
    license="GPLv3",
    zip_safe=False,
    keywords='vad',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPLv3 License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
)
