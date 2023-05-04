from setuptools import find_packages, setup
setup(
    name='pylib',
    packages=find_packages(include=['pylib']),
    version='0.1.0',
    description='Siamese Network Library',
    author='Me',
    license='MIT',
    install_requires=['numpy','pandas','opencv-python','tensorflow','keras','scikit-learn'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
