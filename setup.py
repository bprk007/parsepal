from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace('\n','') for req in requirements] 

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
setup(
name='matproject',
version='0.1',
author='Bhavesh Pareek',
author_email='bprkwork@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')


)