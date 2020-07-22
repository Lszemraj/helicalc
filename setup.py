import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="helicalc",
    version="0.0.0",
    author="Cole Kampa",
    author_email="ckampa13@gmail.com",
    description="Biot-Savart integration for helical solenoids.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FMS-Mu2e/helicalc",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy", "scipy", "pandas", "matplotlib"
    ],
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False,)
