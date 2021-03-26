# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beecolpy",
    version="2.3.2",
    license='MIT',
    author="Samuel Carlos Pessoa Oliveira",
    author_email="samuelcpoliveira@gmail.com",
    description="Artificial Bee Colony solver",
    keywords = ['PSO', 'ABC', 'Bee', 'Colony', 'Binary', 'Angle Modulated', 'Solver', 'Optimize', 'metaheuristic'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renard162/BeeColPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy'],
    python_requires='>=3.0'
)
