from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='fsxlt',
    version='0.0.1',
    description='Few shot',
    license="Apache",
    long_description=long_description,
    long_description_context_type='text/markdown',

    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    python_requires='>=3.9',
)
