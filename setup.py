
import subprocess
from setuptools import setup
from pybind11 import get_cmake_dir

from cmake_build import CMakeBuild, CMakeExtension


rev = '+' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'
                                     ]).decode('ascii').rstrip()
version = "1.3.2" + rev

cpp_module = CMakeExtension(
      name="ffrecord._ffrecord_cpp",
      sourcedir="ffrecord/src",
      cmake_args=[f"-DPYBIND11_CMAKE_DIR={get_cmake_dir()}"]
)

setup(cmdclass={"build_ext": CMakeBuild},
      name='ffrecord',
      version=version,
      description='Fileflyer Record file format',
      author='HFAiLab',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['torch>=1.6', 'pybind11>=2.8', 'tqdm'],
      packages=['ffrecord', 'ffrecord/torch'],
      ext_modules=[cpp_module]
)
