import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nayzakflow", # Replace with your own username
    version="1.0.0",
    author="Nayzak_Team",
    description="Deep Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ahmed-Salah6011/NayzakFlow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
         "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)