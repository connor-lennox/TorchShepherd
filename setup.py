from setuptools import _install_setup_requires, setup, find_packages

setup(
    name="torchshepherd",
    version='0.1.0',
    packages=find_packages(include=['torchshepherd', 'torchshepherd.*']),
)
