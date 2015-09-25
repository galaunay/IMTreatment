from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import shutil

 
# submodules names
submodule_names = ['boundary_layer',
                   'field_treatment',
                   'file_operation',
                   'pod',
                   'tools',
                   'vortex_creation',
                   'vortex_detection'] 

# set extensions
extensions = [Extension("core", ["core.py"], include_dirs=(".",))]
for name in submodule_names:
    path = r"./{0}/{0}.py".format(name)
    ext = Extension(name, [path], include_dirs=('.',))
    extensions.append(ext)

# Distutils
setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = cythonize(extensions),

)


# move *.pyd to the good folder
for name in submodule_names:
    shutil.move('./{0}.pyd'.format(name), r"./{0}")
