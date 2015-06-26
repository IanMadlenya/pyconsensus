#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="pyconsensus",
    version="0.5.8",
    description="Standalone implementation of Augur's consensus mechanism",
    author="Jack Peterson, Joey Krug, Paul Sztorc",
    author_email="<jack@tinybike.net>",
    maintainer="Jack Peterson",
    maintainer_email="<jack@tinybike.net>",
    license="GPL",
    url="https://github.com/AugurProject/pyconsensus",
    download_url = "https://github.com/AugurProject/pyconsensus/tarball/0.5.8",
    packages=["pyconsensus"],
    install_requires=["numpy", "pandas", "six", "weightedstats"],
    keywords = ["consensus", "prediction market", "PM", "truthcoin", "augur", "oracle"]
)
