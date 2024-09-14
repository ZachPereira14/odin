from setuptools import setup, find_packages

setup(
    name='odin',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'odin-cli=odin.cli:main',  # Ensure this points to the main function in cli.py
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
