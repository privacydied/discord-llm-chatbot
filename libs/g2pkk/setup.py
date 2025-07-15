from setuptools import setup, find_packages

setup(
    name='g2pkk',
    version='0.1.2+local',
    packages=find_packages(),
    description='A patched, local version of g2pkk to prevent runtime pip installs.',
    # This package is not meant to be distributed
)
