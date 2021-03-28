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
  exports from several file formats such as DICOM, PNG, and JPEG are provided
  using external libraries (see below).

See also \ref comparison_with_other_libraries "Comparison with other libraries".

TNL also provides several optional components:
<a name="optional-components"></a>

- TNL header files in the
  [src/TNL](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/TNL)
  directory.
- Various pre-processing and post-processing tools in the
  [src/Tools](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Tools)
  directory.
- Python bindings and scripts in the
  [src/Python](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Python)
  directory.
- Examples of various numerical solvers in the
  [src/Examples](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Examples)
  directory.
- Benchmarks in the
  [src/Benchmarks](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Benchmarks)
  directory.

These components can be individually enabled or disabled and installed by a
convenient `install` script. See the [Installation](#installation) section for
details.

## Installation   {#installation}

You can either download the [stable version](http://tnl-project.org/download/)
or directly clone the git repository via HTTPS:

    git clone https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev.git

or via SSH:

    git clone gitlab@mmg-gitlab.fjfi.cvut.cz:tnl/tnl-dev.git

Since TNL is a header-only library, no installation is necessary to actually use
the library in your project. See the [Usage](#usage) section for details.

You may also want to install some optional dependencies and/or compile and
install various tools and examples. See the following section for details.

### Dependencies   {#dependencies}

In order to use TNL, you need to install a compatible compiler, a parallel
computing platform, and (optionally) some libraries.

- __Supported compilers:__
  You need a compiler which supports the [C++14](
  https://en.wikipedia.org/wiki/C%2B%2B14) standard, for example [GCC](
  https://gcc.gnu.org/) 5.0 or later or [Clang](http://clang.llvm.org/) 3.4 or
  later.

- __Parallel computing platforms:__
  TNL can be used with one or more of the following platforms:
    - [OpenMP](https://en.wikipedia.org/wiki/OpenMP) -- for computations on
      shared-memory multiprocessor platforms.
    - [CUDA](https://docs.nvidia.com/cuda/index.html) 9.0 or later -- for
      computations on Nvidia GPUs.
    - [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) -- TNL can
      a library implementing the MPI-3 standard for distributed computing (e.g.
      [OpenMPI](https://www.open-mpi.org/)). For distributed CUDA computations,
      the library must be [CUDA-aware](
      https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/).

- __Libraries:__
  Various libraries are needed to enable optional features or enhance the
  functionality of some TNL components. Make sure that all relevant packages are
  installed and use the appropriate flags when compiling your project.

  <table>
  <tr><th>Library</th>
      <th>Affected components</th>
      <th>Compiler flags</th>
      <th>Notes</th>
  </tr>
  <tr><td> [zlib](http://zlib.net/) </td>
      <td> \ref TNL::Meshes::Readers "XML-based mesh readers" and \ref TNL::Meshes::Writers "writers" </td>
      <td> `-DHAVE_ZLIB -lz` </td>
      <td> </td>
  </tr>
  <tr><td> [TinyXML2](https://github.com/leethomason/tinyxml2/) </td>
      <td> \ref TNL::Meshes::Readers "XML-based mesh readers" </td>
      <td> `-DHAVE_TINYXML2 -ltinyxml2` </td>
      <td> If TinyXML2 is not found as a system library, the `install` script
           will download, compile and install TinyXML2 along with TNL. </td>
  </tr>
  <tr><td> [Metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) </td>
      <td> `tnl-decompose-mesh` </td>
      <td> </td>
      <td> Only used for the compilation of the `tnl-decompose-mesh` tool. </td>
  </tr>
  <tr><td> [libpng](http://www.libpng.org/pub/png/libpng.html) </td>
      <td> \ref TNL::Images "Image processing" classes </td>
      <td> `-DHAVE_PNG_H -lpng` </td>
      <td> </td>
  </tr>
  <tr><td> [libjpeg](http://libjpeg.sourceforge.net/) </td>
      <td> \ref TNL::Images "Image processing" classes </td>
      <td> `-DHAVE_JPEG_H -ljpeg` </td>
      <td> </td>
  </tr>
  <tr><td> [DCMTK](http://dicom.offis.de/dcmtk.php.en) </td>
      <td> \ref TNL::Images "Image processing" classes </td>
      <td> `-DHAVE_DCMTK_H -ldcm...` </td>
      <td> </td>
  </tr>
  </table>

- __Other language toolchains/interpreters:__
    - Python – install an interpreter for using the Python scripts from TNL and
      the corresponding development package (depending on your operating system)
      for building the Python bindings.

### Optional components

TNL provides several optional components such as pre-processing and
post-processing tools which can be compiled and installed by executing the
`install` script:

    ./install

[CMake](https://cmake.org/) 3.13 or later is required for the compilation.

The script compiles and/or installs all [optional components](#optional-components)
into the `~/.local/` directory, and compiles and executes all unit tests from
the [src/UnitTests](
https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/UnitTests)
directory.

Individual components can be disabled and the installation prefix can be changed
by passing command-line arguments to the install script. Run `./install --help`
for details.

## Usage   {#usage}

TNL can be used with various build systems if you configure the compiler flags
as explained below. See also an [example project](
https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/example-project) providing a simple
`Makefile`.

### C++ compiler flags

- Enable the C++14 standard: `-std=c++14`
- Configure the include path: `-I /path/to/include`
    - If you installed TNL with the install script, the include path is
      `<prefix>/include`, where `<prefix>` is the installation path (it is
      `~/.local` by default).
    - If you want to include from the git repository directly, you need to
      specify two include paths: `<git_repo>/src` and `<git_repo/src/3rdparty`,
      where `<git_repo>` is the path where you have cloned the TNL git
      repository.
    - Instead of using the `-I` flag, you can set the `CPATH` environment
      variable to a colon-delimited list of include paths. Note that this may
      affect the build systems of other projects as well. For example:

          export CPATH="$HOME/.local/include:$CPATH"

- Enable optimizations: `-O3 -DNDEBUG` (you can also add
  `-march=native -mtune=native` to enable CPU-specific optimizations).
- Of course, there are many other useful compiler flags. See, for example, our
  [CMakeLists.txt](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/-/blob/develop/CMakeLists.txt)
  file for flags that we use when developing TNL (there are flags for e.g.
  hiding some useless compiler warnings).

### Compiler flags for parallel computing

To enable parallel computing platforms in TNL, additional compiler flags are
needed. They can be enabled by defining a corresponding C preprocessor macro
which has the form `HAVE_<PLATFORM>`, i.e.:

- `-D HAVE_OPENMP` enables OpenMP (also `-fopenmp` is usually needed to enable
  OpenMP support in the compiler)
- `-D HAVE_CUDA` enables CUDA (the compiler must actually support CUDA, use e.g.
  `nvcc` or `clang++`)
    - For `nvcc`, the following experimental flags are also required:
      `--expt-relaxed-constexpr --expt-extended-lambda`
- `-D HAVE_MPI` enables MPI (use a compiler wrapper such as `mpicxx` or link
  manually against the MPI libraries)

### Environment variables

If you installed some TNL tools or examples using the `install` script, we
recommend you to configure several environment variables for convenience. If you
used the default installation path `~/.local/`:

- `export PATH=$PATH:$HOME/.local/bin`
- If TinyXML2 was installed by the `install` script and not as a system package,
  also `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib`

These commands can be added to the initialization scripts of your favourite
shell, e.g. `.bash_profile`.
