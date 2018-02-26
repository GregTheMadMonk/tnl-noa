[![pipeline status](https://jlk.fjfi.cvut.cz/gitlab/mmg/tnl-dev/badges/develop/pipeline.svg)](https://jlk.fjfi.cvut.cz/gitlab/mmg/tnl-dev/commits/develop)

Installation
============

    Requirements:

    To install TNL, you need:

    cmake 3.4 or later (https://cmake.org/download/)
    GNU g++ 4.8 or later (https://gcc.gnu.org/)
    CUDA 8.0 or later (https://developer.nvidia.com/cuda-downloads)

    For image processing problems, you may optionally install:
    DCMTK (http://dicom.offis.de/dcmtk.php.en)
    libpng (http://www.libpng.org/pub/png/libpng.html)
    libjpeg (http://libjpeg.sourceforge.net/)

    The latest release of TNL can be downloaded as:

    wget tnl-project.org/data/src/tnl-0.1.tar.bz2

    Unpack it as:

    tar xvf tnl-0.1.tar.bz2
    cd tnl-0.1

    Executing command

    ./install

    will install TNL to a folder ${HOME}/.local . You may change it by

    ./install --prefix=<TNL prefix>

    During the installation, TNL fetches latest version of Gtest and install it only 
    locally to sub-folders Debug and Release. At the end of the installation, the
    script is checking if the prefix folder is visible to your bash and your linker.
    If not, it informs you how to change your ${HOME}/.bashrc file to fix it.

How to write a simple solver
============================

To implement your own solver:

    Create and go to your working directory

    mkdir MyProblem
    cd Myproblem

    Execute a command tnl-quickstart

    tnl-quickstart

    Answer the questions as, for example, follows

    TNL Quickstart -- solver generator
    ----------------------------------
    Problem name:My Problem
    Problem class base name (base name acceptable in C++ code):MyProblem
    Operator name:Laplace

    Write your numerical scheme by editing a file

    Laplace_impl.h

    on lines:
        34, 141 and 265 for 1D, 2D and 3D problem respectively with explicit time discretization
        101, 211 and 332 for 1D, 2D and 3D problem respectively with (semi-)implicit time discretization
    Compile the program by executing

    make

    for CPU version only or 
   
    make WITH_CUDA=yes

    for a solver running on both CPU and GPU. Run it on your favourite HW architecture by executing

    ./MyProblem

    and following the printed help.
