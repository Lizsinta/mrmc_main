from setuptools import setup, find_packages
setup(
    name='mrmc_package',
    version='1.0',
    author='Lizsinta',
    packages=find_packages(),
    license='MIT Licpip ense',
    platforms='any',
    install_requires=['scipy', 'numpy', 'ase', 'pyqtgraph']
)