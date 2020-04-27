from distutils.core import setup

setup(
  name = 'beecolpy',
  packages = ['beecolpy'],
  version = '1.0',
  license='MIT',
  description = 'Simple Artificial Bee Colony solver',
  author = 'Samuel Carlos Pessoa Oliveira',
  author_email = 'samuelcpoliveira@gmail.com',
  url = 'https://github.com/renard162/BeeColPy',
  download_url = 'https://github.com/renard162/BeeColPy/archive/v1.0.tar.gz',
  keywords = ['PSO', 'ABC', 'Bee', 'Colony', 'Solver', 'Optimize', 'metaheuristic'],
  install_requires=[
          'numpy'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
)
