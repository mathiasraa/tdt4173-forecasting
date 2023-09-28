from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["cloudml-hypertune"]

setup(
    name="hptuning",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Auto MPG XGBoost HP tuning training application",
)
