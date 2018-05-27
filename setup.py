

from setuptools import find_packages, setup

setup(
    name='hypothetical',
    version='0.1.0',
    author='Aaron Schlegel',
    author_email='aaron@aaronschlegel.com',
    description=('Hypothesis testing and other testing methods.'),
    packages=find_packages(exclude=['docs', 'notebooks', 'tests*']),
    include_package_data=True,
    long_description=open('README.md').read(),
    install_requires=['numpy>=1.13.0', 'numpy_indexed>=0.3.5', 'pandas>=0.22.0', 'scipy>=1.1.0'],
    home_page='',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)