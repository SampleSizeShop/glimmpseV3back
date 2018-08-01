from setuptools import setup, find_packages

setup(
    name='demoappback',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['flask', 'flask_cors', ],
    setup_requires=[],
    tests_require=[],
)