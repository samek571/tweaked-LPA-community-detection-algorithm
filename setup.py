import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "lpkit._cprop",
        sources=["src/lpkit/_cprop.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    )
)
