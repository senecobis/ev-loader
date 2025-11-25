from setuptools import setup, find_packages

with open('requirement.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ev-loader",
    version="0.1.0",  # Update version as needed
    author="Your Name",
    author_email="rpellerito@ifi.uzh.ch",
    description="A Python package for loading datasets",
    long_description=open("README.md").read(),
    url="https://github.com/senecobis/ev-loader",  # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.5",  # Adjust as needed
    install_requires=requirements,  # Add dependencies if required
)
