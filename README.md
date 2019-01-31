[![pipeline status](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/badges/develop/pipeline.svg)](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/commits/develop)

# Template Numerical Library

TNL is a collection of building blocks that facilitate the development of
efficient numerical solvers. It is implemented in C++ using modern programming
paradigms in order to provide flexible and user friendly interface. TNL provides
native support for modern hardware architectures such as multicore CPUs, GPUs,
and distributed systems, which can be managed via a unified interface.

Similarly to the STL, features provided by the TNL can be grouped into
several modules:

- _Core concepts_.
  The main concept used in the TNL is the `Device` type which is used in most of
  the other parts of the library. For data structures such as `Array` it
  specifies where the data should be allocated, whereas for algorithms such as
  `ParallelFor` it specifies how the algorithm should be executed.
- _[Containers][containers]_.
  TNL provides generic containers such as array, multidimensional array or array
  views, which abstract data management on different hardware architectures.
- _Linear algebra._
  TNL provides generic data structures and algorithms for linear algebra, such
  as [vectors][vectors], [sparse matrices][matrices],
  [Krylov solvers][linear solvers] and [preconditioners][preconditioners].
   - Sparse matrix formats: CSR, Ellpack, Sliced Ellpack, tridiagonal,
     multidiagonal
   - Krylov solvers: CG, BiCGstab, GMRES, CWYGMRES, TFQMR
   - Preconditioners: Jacobi, ILU(0) (CPU only), ILUT (CPU only)
- _[Meshes][meshes]_.
  TNL provides data structures for the representation of structured or
  unstructured numerical meshes.
- _Solvers for differential equations._
  TNL provides a framework for the development of ODE or PDE solvers.
- _[Image processing][image processing]_.
  TNL provides structures for the representation of image data. Imports and
  exports from several file formats are provided using external libraries, such
  as [DCMTK](http://dicom.offis.de/dcmtk.php.en) for DICOM files,
  [libpng](http://www.libpng.org/pub/png/libpng.html) for PNG files, or
  [libjpeg](http://libjpeg.sourceforge.net/) for JPEG files.

[containers]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Containers.html
[vectors]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/classTNL_1_1Containers_1_1Vector.html
[matrices]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Matrices.html
[linear solvers]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Solvers_1_1Linear.html
[preconditioners]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Solvers_1_1Linear_1_1Preconditioners.html
[meshes]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Meshes.html
[image processing]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Images.html

For more information, see the [full documentation][full documentation].

## Installation

You can either download the [stable version](http://tnl-project.org/download/)
or directly clone the git repository via HTTPS:

    git clone https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev.git

or via SSH:

    git clone gitlab@mmg-gitlab.fjfi.cvut.cz:tnl/tnl-dev.git

Since TNL is a header-only library, no installation is necessary to actually use
the library. You can just extract the [src/TNL](
https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/TNL)
subdirectory, for example into `~/.local/include/`, and configure your compiler
to include headers from this path.

Optionally, you may want to compile and install various [tools and examples](
#optional-components).

### Supported compilers

You need a compiler which supports the [C++14](
https://en.wikipedia.org/wiki/C%2B%2B14) standard, for example [GCC](
https://gcc.gnu.org/) 5.0 or later or [Clang](http://clang.llvm.org/) 3.4 or
later. For CUDA support, you also need [CUDA](
https://docs.nvidia.com/cuda/index.html) 9.0 or later.

### Optional components

TNL provides several optional components such as pre-processing and
post-processing tools which can be compiled and installed by executing the
`install` script:

    ./install

[CMake](https://cmake.org/) 3.12.2 or later is required for the compilation. The
script compiles and/or installs the following components into the `~/.local/`
directory:

- TNL header files from the
  [src/TNL](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/TNL)
  directory.
- Various pre-processing and post-processing tools from the
  [src/Tools](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Tools)
  directory.
- Python bindings and scripts from the
  [src/Python](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Python)
  directory.
- Examples of various numerical solvers from the
  [src/Examples](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Examples)
  directory.
- Benchmarks from the
  [src/Benchmarks](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Benchmarks)
  directory.
- Compiles and executes the unit tests from the
  [src/UnitTests](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/UnitTests)
  directory.

Individual components can be disabled and the installation prefix can be changed
by passing command-line arguments to the install script. Run `./install --help`
for details.

## Documentation

See the [full documentation][full documentation] for the API reference manual,
tutorials and other documented topics.

[full documentation]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/

## Authors

Project leader: Tomáš Oberhuber

Main developers:
- Tomáš Oberhuber
- Jakub Klinkovský

Current developers:
- Aleš Wodecki – adaptive grids
- Jan Schafer – solvers for compressible Navier–Stokes equations
- Matouš Fencl – solvers for the Hamilton–Jacobi equation
- Lukáš Čejka – sparse matrix formats
- Vojtěch Legler – expression templates

Former developers:
- Vít Hanousek – distributed numerical grids
- Vítězslav Žabka – unstructured numerical meshes
- Tomáš Sobotík – solvers for the Hamilton–Jacobi equation
- Libor Bakajsa – sparse matrix formats
- Ondřej Székely – solvers for parabolic problems
- Vacata Jan – sparse matrix formats
- Heller Martin – sparse matrix formats
- Novotný Matěj – high precision arithmetics

See also the [full list of authors and their contributions](
https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/graphs/develop).

## License

Template Numerical Library is provided under the terms of the [MIT License](
https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/blob/develop/LICENSE).
