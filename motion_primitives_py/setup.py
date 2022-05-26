from setuptools import setup

setup(name='motion_primitives_py', version='0.0.1', packages=['motion_primitives_py'], python_requires='>3.6',
      install_requires=['ujson',
                        'sympy',
                        'matplotlib >=3.3.3',
                        'numpy',
                        'pytest',
                        'PyYAML',
                        'scipy',
                        'cvxpy',
                        'reeds_shepp @ git+https://git@github.com/ghliu/pyReedsShepp#egg=reeds_shepp',
                        'sobol_seq @ git+https://git@github.com/naught101/sobol_seq@v0.2.0#egg=sobol_seq'
                        ]
      )
