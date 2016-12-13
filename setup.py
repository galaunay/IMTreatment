from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
# from Cython.Distutils import build_ext


# For cython (to update)
# # submodules names
# submodule_names = ['boundary_layer',
#                    'field_treatment',
#                    'file_operation',
#                    'force_directed_algorithm',
#                    'pod',
#                    'protocols',
#                    'vortex_creation',
#                    'vortex_criterions',
#                    'vortex_detection',
#                    ]
# # set extensions
# extensions = []
# for name in submodule_names:
#     path = r"./IMTreatment/{0}/{0}.py".format(name)
#     ext = Extension(name, [path], include_dirs=('.',))
#     extensions.append(ext)
# extensions.append(Extension('utils', ['units.py', 'codeinteraction.py',
#                                       'files.py',
#                                       'multithreading.py', 'plot.py',
#                                       'progresscounter.py', 'types.py'],
#                             include_dirs=(".",)))
# extensions.append(Extension("core", ["core.py"], include_dirs=(".",)))

# Distutils
setup(
    name='IMTreatment',
    version='0.1',
    description='Tools to analyze PIV data',
    author='Gaby Launay',
    author_email='gaby.launay@tutanota.com',
    packages=['IMTreatment'],
)
