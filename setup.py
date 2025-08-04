from setuptools import setup, find_packages

setup(
    name="mindforge",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mindforge = mindforge.main:main",
        ],
    },
)
