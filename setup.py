from setuptools import find_packages,setup # type: ignore
from typing import List



def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''

    requirement_1st:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e.
                if requirement and requirement!= '-e.':
                    requirement_1st.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    return requirement_1st


setup(
    name="SQLAIproject",
    version="0.0.1",
    author="chandrahas",
    author_email="chandrahasreddyv@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)