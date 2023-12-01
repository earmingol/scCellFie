from setuptools import setup, find_packages

setup(
    name='scCellFie',
    version='0.1.0',
    author='Erick Armingol',
    author_email='erickarmingol@gmail.com',
    description="A tool for studying metabolic tasks from single-cell and spatial transcriptomics",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/earmingol/scCellFie',
    packages=find_packages(),
    install_requires=[
        'scanpy',
        'numpy',
        'pandas',
        'cobra',
        'tqdm',
        'scipy',
        'anndata'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)