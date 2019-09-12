from setuptools import setup, find_packages

#REQUIRED_PACKAGES = ["torch==0.4", "gym", "numpy", "visdom"]

setup(name='apex',
      version='1.0',
      description='Deep reinforcement learning for continuous control',
      author='Pedro Morais, Yesh Godse',
      author_email='autranemorais@gmail.com, yesh.godse@gmail.com',
      license='MIT',
      packages=find_packages(exclude=("tests")),
      #install_requires=REQUIRED_PACKAGES
)
