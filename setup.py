from distutils.core import setup
from pathlib import Path

import setuptools

REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"


def load_requirements(path: Path) -> list:
    with path.open() as f:
        return [
            line.split("#")[0].strip()
            for line in f.read().split("\n")
            if line.split("#")[0].strip()
        ]


requirements = load_requirements(REQUIREMENTS_PATH)


with open("README.md") as f:
    long_description = f.read()


setup(
    name="invokeai_tools",
    version="1.1.0",
    author="Martin Kristiansen",
    author_email="lille.kemiker@gmail.com",
    url="https://github.com/lillekemiker/invokeai_tools",
    packages=setuptools.find_packages(exclude=["tests"]),
    package_data={"": ["py.typed"]},
    include_package_data=True,
    description=long_description.split("\n", 1)[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
