import subprocess
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_py import build_py

import sysconfig


rev = '+' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'
                                     ]).decode('ascii').rstrip()
version = "1.2.0" + rev

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-DNDEBUG", "-O3"]
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra"]

ext_modules = [
    Extension(
        "_ffrecord_cpp",
        sources=sorted(glob("ffrecord/src/*.cpp")),
        include_dirs=['ffrecord/src'],
        language='c++',
        extra_compile_args=extra_compile_args,
        libraries=['aio', 'z'],
    ),
]

setup(cmdclass={"build_py": build_py},
      name='ffrecord',
      version=version,
      keywords='Fileflyer Record file format',
      author='HFAiLab',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['libaio', 'torch>=1.6'],
      packages=['ffrecord', 'ffrecord/torch'],
      ext_modules=ext_modules
)
