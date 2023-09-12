from setuptools import setup, find_packages

setup(
    name="stein-correction",
    version="0.0.1",
    description="Fast Stein correction for discrete data.",
    author="Ryosuke Shibukawa",
    author_email="ryosuke.shibukawa@gmail.com",
    url="https://github.com/tsudalab/fast-stein-correction",
    packages=find_packages(include=["stein"]),
    install_requires=[
        "cvxopt==1.3.2",
        "dimod==0.12.12",
        "dwave-cloud-client==0.10.6",
        "dwave-neal==0.6.0",
        "dwave-preprocessing==0.6.3",
        "dwave-samplers==1.1.0",
        "dwave-system==1.20.0",
        "matplotlib",
        "PyYAML",
        "papermill",
        "pytest==7.4.1",
    ],
    tests_require=["pytest"],
)
