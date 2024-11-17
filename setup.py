from setuptools import setup

setup(
    name='bisetar',
    version='0.9.1',
    description='MCMC for Bayesian Bidirectional SETAR model',
    url='https://github.com/wilson-ye-chen/bisetar',
    author='Wilson Ye Chen',
    license='MIT',
    packages=['bisetar'],
    install_requires=['numpy', 'scipy', 'statsmodels']
    )
