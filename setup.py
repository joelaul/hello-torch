from setuptools import setup, find_packages

# setuptools is one example of a python build "backend"

setup(
    name="hello-pytorch",
    version="0.1.0",
    packages=find_packages(),  # Automatically discover all packages and sub-packages
    install_requires=[
        "numpy>=1.19.2",
        "requests>=2.25.1"
    ],
    entry_points={
        'console_scripts': [
            'my_command=my_package.module1:main_function',
        ],
    },
    author="Your Name",
    description="A brief description of your project",
    url="https://github.com/yourusername/my_project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
