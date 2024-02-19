from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='Wav2TextGrid',
    version='0.0.11',
    author="Prad Kadambi",
    description="A python forced alignment package",
    author_email="pkadambi@asu.edu",
    packages=find_packages(),
    install_requires=required_packages,
    classifiers=['Programming Language :: Python :: 3', 'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent'],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'wav2textgrid = Wav2TextGrid.wav2textgrid:main'
        ]
    }
)