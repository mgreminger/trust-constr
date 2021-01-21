import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trust-constr",
    version="0.0.1",
    description="trust-constr optimization algorithm from the SciPy project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mgreminger/trust-constr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['numpy>=1.16.5']
)