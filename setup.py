
# setup.py
from setuptools import setup, find_packages

setup(
    name="python_slopes",
    version="0.1.0",
    packages=find_packages(where="src"),  # Key change: Look in 'src'
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0"
    ],
    author="Griffin Berse",
    author_email="griffinkberse@gmail.com",
    description="A slope stability analysis tool using OMS and Bishop methods",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)