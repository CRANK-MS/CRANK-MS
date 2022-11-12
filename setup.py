import setuptools

setuptools.setup(
    name = 'crank-ms',
    version = '0.0.1',
    author = 'Chonghua Xue',
    author_email = 'cxue2@bu.edu',
    url = 'https://github.com/CRANK-MS/CRANK-MS',
    # description = 'Re-implementation of Python Speech Features Extraction on CUDA.',
    packages = setuptools.find_packages(),
    python_requires = '>=3.7',
    classifiers=[
        'Environment :: GPU :: NVIDIA CUDA',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'sklearn',
    ],
)