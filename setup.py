from setuptools import setup
from setuptools.command.build_py import build_py

import subprocess

rev = '+' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'
                                     ]).decode('ascii').rstrip()
version = "1.1.0" + rev

setup(cmdclass={
    "build_py": build_py,
},
      name='ffrecord',
      version=version,
      keywords='Fileflyer Record file format',
      author='HFAiLab',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['libaio', 'torch>=1.6'],
      packages=['ffrecord', 'ffrecord/torch'])
