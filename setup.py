from setuptools import setup, find_packages

setup(name='rlbase',
    version='0.1',
    description='Lightweight Pytorch reinforcement learning library',
    author='Pedro Morais',
    author_email='autranemorais@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['pytorch', 'numpy', 'bokeh'],
    keywords=['deep-learning', 'reinforcement-learning', 'machine-learning',
            'pytorch', 'ppo'],
    python_requires='>=3',
    zip_safe=False)
