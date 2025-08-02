from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str) -> List[str]:
    """
    This function reads the requirements from a file and returns a list of packages.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    requirements = [req.replace("\n","") for req in requirements if  not req.startswith('-e')]
    return requirements





setup(
    name='project_1',
    version='0.0.1',
    author='Sai Chandana',
    author_email="sayachandana@gmail.com",
    description='A sample project',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)