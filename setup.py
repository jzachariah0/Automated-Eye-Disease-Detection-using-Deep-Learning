from setuptools import setup, find_packages

setup(
    name='Eye Disease Detection Program',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.6.0',
        'matplotlib==3.4.3',
        'numpy==1.21.2',
        'scikit-learn==0.24.2',
        'scipy==1.7.1',
    ],
    entry_points={
        'console_scripts': [
            'eye_disease_detection = app:main'
        ]
    },
)
