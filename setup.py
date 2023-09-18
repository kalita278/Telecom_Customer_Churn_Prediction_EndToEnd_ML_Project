from setuptools import find_packages, setup
from typing import List


hyphen='-e .'
def get_requirements(path:str) ->List[str]:
    requirement =[]

    with open(path) as a:
        requirements=a.readlines()
        for i in requirements:
            requirements=[i.replace("\n","")]
        
        if hyphen in requirements:
            requirements.remove(hyphen)
        return requirements
    

setup(
    name="Telecom customer churn prdiction end to end ml project",
    version='0.0.1',
    author="Mrinal Kalita",
    author_email="kalita278@gmailcom",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)