import sys
import setuptools
from setuptools import find_packages


def read_requirements(requirements_file_path):
    requirements_file = open(requirements_file_path, 'r')
    requirements = requirements_file.readlines()
    requirements_file.close()
    requirements = [r.strip() for r in requirements]
    return requirements

def read_version(version_file_path):
    version_file = open(version_file_path, 'r')
    version = version_file.readlines()
    version_file.close()

    for i, v in enumerate(version):
        if v.strip().startswith('#'):
            continue
        if "__version__" in v:
            version = version[i].split("=")[1].strip("\n").strip("'").strip('"')
            break

    return version

sys_version = sys.version_info
python_version_major = sys.version_info.major
python_version_minor = sys.version_info.minor

supported_python_major_versions = [3]
supported_python_minor_versions = [8,10]

if python_version_major in supported_python_major_versions \
    and python_version_minor in supported_python_minor_versions:
    if python_version_minor == 10:
        requirements_path = "./requirements.txt"
    else:
        requirements_path = f"./version_specific_requirements/requirements_py_{python_version_major}_{python_version_minor}.txt"
    basic_requirements = read_requirements(requirements_path)
else:
    raise Exception(f"Module not compatible with Python{str(python_version_major)+'.'+str(python_version_minor)}")

setuptools.setup(
    name='nerf_sandbox',
    version=read_version("nerf_sandbox/version.py"),
    description='Package containing python implementations of NeRF MLPs and training code',
    long_description='Package containing python implementations of NeRF MLPs and training code suitable for experimentation',
    long_description_content_type='text/markdown',
    keywords='common',
    packages=find_packages(exclude='configs'),
    classifiers=[
        'Operating System :: Linux',
        f'Programming Language :: Python :: {str(python_version_major)+"."+str(python_version_minor)}',
    ],
    license='private',
    zip_safe=False,
    install_requires=basic_requirements,
    package_data={
        "": ["*.json", "*.pbtxt"],
    }
)

