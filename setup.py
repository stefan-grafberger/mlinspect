"""
For 'python setup.py develop' and 'python setup.py test'
"""
import os
from setuptools import setup, find_packages

ROOT = os.path.dirname(__file__)

with open(os.path.join(ROOT, "requirements", "requirements.txt"), encoding="utf-8") as f:
    required = f.read().splitlines()

with open(os.path.join(ROOT, "requirements", "requirements.dev.txt"), encoding="utf-8") as f:
    test_required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlinspect",
    version="0.0.1.dev0",
    description="Inspect ML Pipelines in the form of a DAG",
    author='Stefan Grafberger',
    author_email='stefangrafberger@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    tests_require=test_required,
    extras_require={'dev': test_required},
    license='Apache License 2.0',
    python_requires='==3.10.*',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10'
    ]
)
