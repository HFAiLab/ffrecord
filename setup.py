
import subprocess
from setuptools import setup

from cmake_build import CMakeBuild, CMakeExtension


rev = '+' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'
                                     ]).decode('ascii').rstrip()
version = "1.3.0" + rev

cpp_module = CMakeExtension(name="ffrecord._ffrecord_cpp", sourcedir="ffrecord/src")


setup(cmdclass={"build_ext": CMakeBuild},
      name='ffrecord',
      version=version,
      description='Fileflyer Record file format',
      author='HFAiLab',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['torch>=1.6'],
      packages=['ffrecord', 'ffrecord/torch'],
      ext_modules=[cpp_module]
)
