# Template Numerical Library

![TNL logo](tnl-logo.jpg)

TNL is a collection of building blocks that facilitate the development of
efficient numerical solvers. It is implemented in C++ using modern programming
paradigms in order to provide flexible and user friendly interface. TNL provides
native support for modern hardware architectures such as multicore CPUs, GPUs,
and distributed systems, which can be managed via a unified interface.

Similarly to the STL, features provided by the TNL can be grouped into
several modules:

> TODO: Each topic in this list should have a separate page or tutorial.

- _Core concepts_.
  The main concepts used in TNL are the _memory space_, which represents the
  part of memory where given data is allocated, and the _execution model_,
  which represents the way how given (typically parallel) algorithm is executed.
  For example, data can be allocated in the main system memory, in the GPU
  memory, or using the CUDA Unified Memory which can be accessed from the host
  as well as from the GPU. On the other hand, algorithms can be executed using
  either the host CPU or an accelerator (GPU), and for each there are many ways
  to manage parallel execution. The usage of memory spaces is abstracted with
  \ref TNL::Allocators "allocators" and the execution model is represented by
  \ref TNL::Devices "devices". See the \ref core_concepts "Core concepts" page
  for details.
- \ref TNL::Containers "Containers".
  TNL provides generic containers such as array, multidimensional array or array
  views, which abstract data management and execution of common operations on
  different hardware architectures.
- _Linear algebra._
  TNL provides generic data structures and algorithms for linear algebra, such
  as \ref TNL::Containers::Vector "vectors",
  \ref TNL::Matrices "sparse matrices",
  \ref TNL::Solvers::Linear "Krylov solvers" and
  \ref TNL::Solvers::Linear::Preconditioners "preconditioners".
   - Sparse matrix formats: CSR, Ellpack, Sliced Ellpack, tridiagonal,
     multidiagonal
   - Krylov solvers: CG, BiCGstab, GMRES, CWYGMRES, TFQMR
   - Preconditioners: Jacobi, ILU(0) (CPU only), ILUT (CPU only)
- \ref TNL::Meshes "Meshes".
  TNL provides data structures for the representation of structured or
  unstructured numerical meshes.
- _Solvers for differential equations._
  TNL provides a framework for the development of ODE or PDE solvers.
- \ref TNL::Images "Image processing".
  TNL provides structures for the representation of image data. Imports and
  exports from several file formats are provided using external libraries, such
  as [DCMTK](http://dicom.offis.de/dcmtk.php.en) for DICOM files,
  [libpng](http://www.libpng.org/pub/png/libpng.html) for PNG files, or
  [libjpeg](http://libjpeg.sourceforge.net/) for JPEG files.

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
