from setuptools import setup
import os
import subprocess

# compile pointconvformer
pcf_path = os.path.join(os.path.dirname(__file__), 'pointconvformer')
subprocess.run(['chmod', 'u+x', os.path.join(pcf_path, 'setup.sh')])
subprocess.run(['chmod', 'u+x', os.path.join(pcf_path, 'cpp_wrappers', 'compile_wrappers.sh')])
subprocess.run(['./setup.sh'], cwd=pcf_path)


setup(
	name='PointConvFormer Wrapper',
	version='1.0',
	description='Wrapper code for pointconvformer',
	author='Herbert Wright',
	author_email='u1331078@utah.edu',
	packages=['pointconvformer_wrapper'],
)

