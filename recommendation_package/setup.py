from setuptools import setup, find_packages

setup(
    name='recommender',
    version='0.1.0',
    description='A modular recommendation system with collaborative and content-based filtering',
    author='Luc Will',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'pandas>=1.3',
        'numpy>=1.21',
        'scikit-learn>=1.0'
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
