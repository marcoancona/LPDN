import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lpdn",
    version="0.0.1",
    author="Marco Ancona",
    author_email="marco.ancona@inf.ethz.ch",
    description="Implementation of Lightweight Probabilistic Deep Network (inference-only) for Keras and Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcoancona/LPDN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)