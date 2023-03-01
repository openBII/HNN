from setuptools import setup, find_packages
from os import path


DIR = path.dirname(path.abspath(__file__))
with open(path.join(DIR, 'README.md')) as f:
    README = f.read()

setup(
    name="hnn",
    version="0.0.3.1",
    packages=find_packages(exclude=["examples", "examples.*", "unit_tests", "unit_tests.*"]),
    keywords=["hybrid neural networks", "spiking neural networks", "quantization"],
    description="A programming framework based on PyTorch for hybrid neural networks with automatic quantization",
    long_description=README,
    long_description_content_type='text/markdown',
    license="Apache License 2.0",
    url="https://github.com/openBII/HNN",
    author="Huanyu",
    author_email="huanyu.qu@hotmail.com",
    include_package_data=True,
    platforms="any",
    install_requires=['numpy', 'torch==1.11.0', 'torchvision', 'onnx', 'onnx-simplifier', 'spikingjelly'],
    tests_require=['pytest', 'pytest-html', 'pytest-xdist'],
)
