from setuptools import setup, find_packages

#REQUIRED_PACKAGES = ["torch==0.4", "gym", "numpy", "visdom"]

setup(name='deeprl',
      version='0.1',
      description='Deep reinforcement learning for continuous control',
      author='Pedro Morais',
      author_email='autranemorais@gmail.com',
      license='MIT',
      packages=find_packages(exclude=("cassie", "tests")),
      #install_requires=REQUIRED_PACKAGES
)