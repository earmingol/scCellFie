import ast
import re
from setuptools import setup, find_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('sccellfie/__init__.py', 'rb') as f:
    hit = _version_re.search(f.read().decode('utf-8')).group(1)
    version = str(ast.literal_eval(hit))

setup(
    name='scCellFie',
    version=version,
    author='Erick Armingol',
    author_email='erickarmingol@gmail.com',
    description="A tool for inferring metabolic activities from single-cell and spatial transcriptomics",
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    url='https://github.com/earmingol/scCellFie',
    packages=find_packages(),
    install_requires=[
        'scanpy',
        'numpy',
        'pandas',
        'cobra',
        'tqdm',
        'scipy',
        'anndata',
        'squidpy',
        'networkx',
        'geopandas',
        'esda',
        'matplotlib',
        'seaborn',
        'glasbey'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)