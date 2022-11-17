from setuptools import setup
import os

packages = []
root_dir = os.path.dirname(__file__)
if root_dir:
    os.chdir(root_dir)

for dirpath, dirnames, filenames in os.walk('consumption_model_ch'):
    # Ignore dirnames that start with '.'
    if '__init__.py' in filenames:
        pkg = dirpath.replace(os.path.sep, '.')
        if os.path.altsep:
            pkg = pkg.replace(os.path.altsep, '.')
        packages.append(pkg)



setup(
    name = 'consumption_model_ch',
    version = '0.2.dev1',
    packages = packages,
    python_requires = '>=3.8',
    author = "Aleksandra Kim",
    author_email = "aleksandra.kim@icloud.com",
    description = 'create Swiss household consumption model',
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    url = "https://github.com/aleksandra-kim/consumption_model_ch",
    classifiers =[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        ],
    )
