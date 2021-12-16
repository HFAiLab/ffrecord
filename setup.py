
import subprocess
from setuptools import setup
import sysconfig

from cmake_build import CMakeBuild, CMakeExtension


rev = '+' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'
                                     ]).decode('ascii').rstrip()
version = "1.3.0" + rev

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-DNDEBUG", "-O3"]
extra_compile_args += ["-std=c++11", "-Wall", "-Wextra"]


cpp_module = CMakeExtension(name="_ffrecord_cpp", sourcedir="ffrecord/src")


setup(cmdclass={"build_ext": CMakeBuild},
      name='ffrecord',
      version=version,
      keywords='Fileflyer Record file format',
      author='HFAiLab',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['libaio', 'torch>=1.6'],
      packages=['ffrecord', 'ffrecord/torch'],
      ext_modules=[cpp_module]
)
