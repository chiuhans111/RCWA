from setuptools import setup

setup(
   name='rcwa',
   version='1.0',
   description='RCWA',
   author='Hans Chiu',
   author_email='hanschiu3d@gmail.com',
   packages=['rcwa'],  #same as name
   install_requires=['numpy', 'matplotlib', 'tqdm'], #external packages as dependencies
)
