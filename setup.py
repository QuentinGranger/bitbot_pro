from setuptools import setup, find_packages

setup(
    name="bitbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "plotly>=5.3.0",
        "loguru>=0.7.0",
        "websockets>=10.0",
        "python-binance>=1.0.17",
        "ccxt>=3.0.0",
    ],
    python_requires=">=3.8",
)
