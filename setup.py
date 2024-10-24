from setuptools import setup, find_packages
import re
from pathlib import Path

package_name = "equilib_wrapper"

if __name__ == "__main__":
    here = Path(__file__).parent

    # Read package version
    version = re.search(
        r'__version__ = "(.+?)"',
        (here / package_name / "__init__.py").read_text("utf8"),
    ).group(1)

    # Read requirements from requirements.txt
    requirements = (
        (here / "requirements.txt").read_text("utf8").strip().split("\n")
    )

    setup(
        name=package_name,
        description="Simple wrapper for pyequilib",
        version=version,
        author="Christopher Hassø",
        author_email="christopher@hassoe.dk",
        license="MIT",
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements,
    )