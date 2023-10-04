from setuptools import setup, find_packages

setup(
    version='0.0.31',
    name='app',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['Flask>1.0.0', 'flask_cors', ],
    setup_requires=[],
    tests_require=[],
)