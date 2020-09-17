from setuptools import setup, find_packages, Extension
import io
import os
import re

PKG_NAME = 'embedden'
README_PATH = 'README.md'


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop('encoding', 'utf-8')
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read(PKG_NAME, '__init__.py'),
        re.MULTILINE).group(1)
    return version


def get_long_description():
    return _read(README_PATH)


def get_requirements(path):
    content = _read(path)
    return [
        req
        for req in content.split("\n")
        if req != '' and not req.startswith('#')
    ]


def get_ext_modules():
    from Cython.Build import cythonize
    import numpy as np

    ext_modules = [
        Extension(
            "{}.layout._umap".format(PKG_NAME),
            sources=['{}/layout/_umap.pyx'.format(PKG_NAME)],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "{}.layout._fitsne".format(PKG_NAME),
            sources=[
                '{}/layout/_fitsne.pyx'.format(PKG_NAME),
                "src/fitsne/nbodyfft.cpp",
                "src/fitsne/sptree.cpp",
                "src/fitsne/tsne.cpp"
            ],
            language="c++",
            extra_compile_args=["-std=c++11", "-O3", '-pthread', "-lfftw3", "-lm"],
            extra_link_args=['-lfftw3', '-lm'],
            include_dirs=[np.get_include()],
        )
    ]

    return cythonize(ext_modules)


setup(
    name=PKG_NAME,
    author='Nezar Abdennur',
    author_email='nezar@valent-ai.com',
    version=get_version(),
    license='MIT',
    description='An API layer for embeddings',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/valent-ai/embedden',
    # keywords=[],
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=get_requirements('requirements.txt'),
    # tests_require=tests_require,
    # extras_require=extras_require,
    ext_modules=get_ext_modules()
)
