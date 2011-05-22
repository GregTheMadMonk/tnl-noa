dnl Process this file with autoconf to produce a configure script
VERSION="0.1"
AC_INIT(tnl,0.1)
AC_CONFIG_MACRO_DIR([m4])

dnl automake init
AM_INIT_AUTOMAKE(libtnl,0.1)
AM_CONFIG_HEADER(config.h)
AM_PROG_LIBTOOL




dnl -----------------------------------------
dnl ----------- check for CUDA --------------
dnl -----------------------------------------
AC_ARG_VAR(CUDA_LIBS,
           AS_HELP_STRING([],[point to a directory with CUDA libraries e.g. /usr/local/cuda/lib])
           )
AC_ARG_VAR(CUDA_HEADERS,
           AS_HELP_STRING([],[point to a directory with CUDA headers e.g. /usr/local/cuda/include])
           )
if test x$CUDA_LIBS = x;
then
   CUDA_LIBS="/usr/local/cuda/lib"
fi
if test x$CUDA_HEADERS = x;
then
   CUDA_HEADERS="/usr/local/cuda/include"
fi
AC_ARG_WITH(cuda,
            AS_HELP_STRING([--with-cuda=[yes/no]],[turn on nVidia CUDA support [default=yes]]),
            with_cuda=$withval,
            with_cuda=yes)
AC_ARG_WITH(cuda_headers,
            AS_HELP_STRING([--with-cuda-headers=dir],[nVidia CUDA headers [default=/usr/local/cuda/include]]),
            CUDA_HEADERS=$withval)
AC_ARG_WITH(cuda_libdir,
            AS_HELP_STRING([--with-cuda-libdir],
                           [says where the CUDA libraries can be found, default is /usr/local/cuda/lib]),
            CUDA_LIBS=$withval)
AC_ARG_WITH(cuda_arch,
            AS_HELP_STRING([--with-cuda-arch],
                           [specifies the CUDA architecture, can be 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3 or 3.0 - default is 1.3]),
            CUDA_ARCH=$withval,
            CUDA_ARCH="1.3")

working_nvcc="no"  
if test x$with_cuda = xyes;
then
   STORE_CXXFLAGS=$CXXFLAGS
   STORE_CPPFLAGS=$CPPFLAGS
   STORE_LDFLAGS=$LDFLAGS
   CUDA_LDFLAGS="-L$CUDA_LIBS"
dnl   CUDA_CXXFLAGS="-I$CUDA_HEADERS -DHAVE_CUDA -Xcompiler -std=c++0x"
dnl -DHAVE_CUDA should not be added here but in approrpiate Makefile.am
   CUDA_CXXFLAGS="-I$CUDA_HEADERS"
   CXXFLAGS="$CXXFLAGS -I$CUDA_HEADERS"
   CPPFLAGS="$CPPFLAGS -I$CUDA_HEADERS"
   LDFLAGS="$LDFLAGS $CUDA_LDFLAGS"
   AC_CHECK_HEADER(cuda.h,HAVE_CUDA="yes",HAVE_CUDA="no")
   if test x$HAVE_CUDA = xyes;
   then
      AC_CHECK_HEADER(cuda.h,HAVE_CUDA="yes",HAVE_CUDA="no")
   fi
   if test x$HAVE_CUDA = xyes;
   then
      AC_CHECK_HEADER(cuda_runtime.h,HAVE_CUDA="yes",HAVE_CUDA="no")
   fi
   if test x$HAVE_CUDA = xyes;
   then
      AC_CHECK_LIB(cudart,cudaMalloc,HAVE_CUDA="yes",HAVE_CUDA="no")
   fi
   AC_CHECK_PROG(cuda,nvcc,"yes","no")
   AC_ARG_VAR([NVCC], [nvcc compiler to use])
   AC_PATH_PROG([NVCC], [nvcc], [no])
   working_nvcc=no
   if test "x$NVCC" != "xno";
   then
      AC_MSG_CHECKING([whether nvcc works])
      echo "" >> conftest.cu
      echo " __global__ void test_cuda() { " >> conftest.cu
      echo "  const int tid = threadIdx.x; " >> conftest.cu
      echo "  const int bid = blockIdx.x; " >> conftest.cu
      echo " __syncthreads(); " >> conftest.cu
      echo " }" >> conftest.cu
      
      if $NVCC $CXXFLAGS conftest.cu > /dev/null 2>&1 && test_whether_output_files_are_ok;
      then
         working_nvcc=yes
         AC_SUBST(NVCC)
      fi
      rm -f conftest.cu conftest.o conftest.what conftest.ever
      AC_MSG_RESULT([$working_nvcc])
   fi
   
dnl   if test x$HAVE_CUDA = xyes; # TODO: && test x$working_nvcc = xyes;
dnl   then
   CUDA_LDFLAGS="$CUDA_LDFLAGS -lcudart"
   CC="nvcc"
   CXX="nvcc"
   case "$CUDA_ARCH"  in
         1.0 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_10 -DCUDA_ARCH=10"
         ;;
         1.1 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_11 -DCUDA_ARCH=11"
         ;;
         1.2 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_12 -DCUDA_ARCH=12"
         ;;
         1.3 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_13 -DCUDA_ARCH=13"
         ;;
         2.0 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_20 -DCUDA_ARCH=20"
         ;;
         2.1 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_21 -DCUDA_ARCH=21"
         ;;
         2.2 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_22 -DCUDA_ARCH=22"
         ;;
         2.3 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_23 -DCUDA_ARCH=23"
         ;;
         3.0 )
            CUDA_CXXFLAGS="$CUDA_CXXFLAGS -arch=sm_30 -DCUDA_ARCH=30"
         ;;         
      esac  
   dnl      DBGCXXFLAGS="$DBGCXXFLAGS -deviceemu"    
   dnl   else
   dnl      CUDA_LDFLAGS=""
   dnl      CUDA_CXXFLAGS=""
   dnl   fi
   
   dnl ----------------------------------------
   dnl -------- check for CUSP ----------------
   dnl ----------------------------------------
   dnl Note: CUSP implements sparse matrices in
   dnl CUDA. We use it only for testing.
   AC_ARG_VAR(CUSP_DIR,
              AS_HELP_STRING([],[point to a directory with CUSP and THRUST library e.g. /usr/local])
              )
   AC_ARG_WITH(cusp_dir,
               AS_HELP_STRING([--with-cusp-dir],
                              [specifies the directory where the CUSP and THRUST libraries are installed (set /usr/local if CUSP is in /usr/local/cusp)]),
               WITH_CUSP_DIR=$withval,
               WITH_CUSP_DIR="$CUSP_DIR")
   
   if test x$WITH_CUSP_DIR != x;
   then
      CUSP_CXXFLAGS="-I$WITH_CUSP_DIR -I$WITH_CUSP_DIR/cusp -I$WITH_CUSP_DIR/thrust"
      AC_SUBST(CUSP_CXXFLAGS)
      CUDA_CXXFLAGS="$CUDA_CXXFLAGS -DHAVE_CUSP"
dnl AC_DEFINE(HAVE_CUSP, [1], [Define to 1 if you have the CUSP library is installed on this system.])
   fi 
   
   CXXFLAGS=$STORE_CXXFLAGS
   CPPFLAGS=$STORE_CPPFLAGS
   LDFLAGS=$STORE_LDFLAGS
else
   CXXFLAGS="$CXXFLAGS -std=c++0x"
fi
AM_CONDITIONAL([working_nvcc], [test "x$working_nvcc" = "xyes"])

dnl -----------------------------------------
dnl ----------- check for progs -------------
dnl -----------------------------------------

AC_PROG_CXX
AC_PROG_AWK
AC_LANG_SAVE
AC_LANG_CPLUSPLUS

dnl -----------------------------------------
dnl ----------- check for icpc --------------
dnl -----------------------------------------

if test x"$CXX" = xicpc; then
   CXXFLAGS="$CXXFLAGS -DHAVE_ICPC"
fi

dnl -----------------------------------------
dnl ----------- check for debug--------------
dnl -----------------------------------------
AC_ARG_ENABLE([debug],
              AC_HELP_STRING( [--enable-debug=[no/yes]],[Turn on debugging [default=no] ] ) )
if test "x$enable_debug" = xyes; then
   DBGCXXFLAGS="$DBGCXXFLAGS -O0 -DDEBUG"
   if test x$CXX != xnvcc;
   then
      DBGCXXFLAGS="$DBGCXXFLAGS -g3 -Wall -W -ansi -Wno-unused"
   fi
   
   DBGSTATUS="yes"
else
   DBGSTATUS="no"
fi

AC_SUBST(DBGCXXFLAGS)

OPTIMISECXXFLAGS="-O3"
AC_SUBST(OPTIMISECXXFLAGS)

dnl -----------------------------------------
dnl ----------- check for gprof -------------
dnl -----------------------------------------
AC_ARG_WITH(gprof,
            AS_HELP_STRING([--with-gprof],[compile with gprof support[default=no]]),
            with_gprof=$withval,
            with_gprof=no)
if test x"$with_gprof" = xyes;
then
   CXXFLAGS="$CXXFLAGS -pg"
   LDFLAGS="$LDFLAGS -pg"
fi

dnl -----------------------------------------
dnl ----------- check for gcov --------------
dnl -----------------------------------------
AC_ARG_WITH(gcov,
            AS_HELP_STRING([--with-gcov],[compile with gcov support[default=no]]),
            with_gcov=$withval,
            with_gcov=no)
if test x"$with_gcov" = xyes;
then
   CXXFLAGS="-fprofile-arcs -ftest-coverage"
fi

dnl -----------------------------------------
dnl --------- check for dmalloc -------------
dnl -----------------------------------------

AC_ARG_WITH(dmalloc, [  --with-dmalloc=[no/yes/full]	compile with Debug Malloc Library])
if test x"$with_dmalloc" = xfull;
then
  CXXFLAGS="$CXXFLAGS -DDMALLOC_FUNC_CHECK"
  with_dmalloc=yes
fi
if test x"$with_dmalloc" = xyes;
then
  AC_CHECK_LIB(dmalloc,dmalloc_error,,AC_MSG_ERROR(Unable to find Debug malloc library))
  CXXFLAGS="$CXXFLAGS -DGAG_DEBUG_MEM"
  LIBS="$LIBS -ldmalloc"
fi

dnl -----------------------------------------
dnl ---------- check for malloc -------------
dnl -----------------------------------------


AC_ARG_WITH(malloc, [  --with-efence compile with Electric Fence Library])
if test x"$with_efence" = xyes;
then
  AC_CHECK_LIB(efence,malloc,,AC_MSG_ERROR(Unable to find Electric Fence library))
  LIBS="$LIBS -lefence"
fi

AC_ARG_WITH(gprof, [  --with-gprof compile with gprof extra code])
if test x"$with_gprof" = xyes;
then
   CXXFLAGS="$CXXFLAGS -pg"
fi

AC_CHECK_FUNCS(bzero)
AC_CHECK_FUNCS(pow)
AC_CHECK_FUNCS(sqrt)

AC_C_CONST
AC_C_INLINE
AC_FUNC_ERROR_AT_LINE

dnl check headers
AC_HEADER_STDC
AC_CHECK_HEADERS(float.h)
AC_CHECK_HEADERS(math.h)
AC_CHECK_HEADERS(iostream)
AC_CHECK_HEADERS(fstream)
AC_CHECK_HEADERS(iomanip)
AC_CHECK_HEADERS(sys/resource.h)
AC_CHECK_HEADERS(stddef.h)
AC_CHECK_HEADERS(sys/time.h)




dnl -----------------------------------------
dnl ------------ check for MPI --------------
dnl -----------------------------------------

MPISTATUS="no"
AC_ARG_WITH(mpi,
            AS_HELP_STRING([--with-mpi],[compile with MPI support[default=yes]]),
            with_mpi=$withval,
            with_mpi=yes)
if test x"$with_mpi" == xyes;
then
   AC_CHECK_PROG( MPICXX, mpic++, yes, no )
   if test x"$MPICXX" = xyes;
   then
      MPICXXFLAGS=`mpic++ -showme:compile`
      MPILDFLAGS=`mpic++ -showme:link`
      MPISTATUS="yes"
   else
      AC_CHECK_PROG( MPICC, mpiCC, yes, no )
      if test x"$MPICC" = xyes;
      then
         MPICXXFLAGS="-DUSE_STDARG -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_UNISTD_H=1 -DHAVE_STDARG_H=1 -DUSE_STDARG=1 -DMALLOC_RET_VOID=1"
         MPILDFLAGS="-L/usr/lib/mpich/lib -lpmpich++ -lmpich -lpthread -lrt"
         MPISTATUS="yes"
      fi
   fi
   STORE_CXXFLAGS="$CXXFLAGS"
   
   CXXFLAGS="$CXXFLAGS $MPICXXFLAGS"
   AC_CHECK_HEADERS(mpi.h, MPICXXFLAGS="$MPICXXFLAGS -DHAVE_MPI")
   CXXFLAGS="$STORE_CXXFLAGS"
   AC_SUBST(MPICXXFLAGS)
   AC_SUBST(MPILDFLAGS)
fi

dnl -----------------------------------------
dnl ---------- check for libbzip2 -----------
dnl -----------------------------------------
AC_LANG_PUSH(C)
AC_CHECK_LIB(bz2,BZ2_bzReadOpen,,AC_MSG_ERROR(Unable to find bzip2 library))
LIBS="$LIBS -lbz2"
AC_CHECK_HEADER(bzlib.h,
                BZIP2_CXXFLAGS="-DHAVE_BZIP2",
                AC_MSG_ERROR(Unable to find bzip2 header))
AC_LANG_POP()

dnl -----------------------------------------
dnl ---------- check for CppUnit ------------
dnl -----------------------------------------
AC_CHECK_HEADER(cppunit/TestCase.h)
STORE_LDFLAGS=$LDFLAGS
LDFLAGS="$LDFLAGS -lcppunit"
AC_TRY_LINK([#include <cppunit/TestCase.h>],
            [
               class IntegerTest : public CppUnit::TestCase 
               {  
                  public: 
                  void runTest()
                  { 
                     CPPUNIT_ASSERT( 1 == 1 );
                  };
                }
              ],HAVE_CPPUNIT="yes",HAVE_CPPUNIT="no")
if test x$HAVE_CPPUNIT="xyes";
then
   CXXFLAGS="$CXXFLAGS -DHAVE_CPPUNIT"
else
   LDFLAGS=$STORE_LDFLAGS
fi
AC_LANG_RESTORE

AM_CONDITIONAL([BUILD_CUDA],[test x"$HAVE_CUDA" = xyes] )
AM_CONDITIONAL([BUILD_MPI],[test x"$MPISTATUS" = xyes] )
AM_CONDITIONAL([BUILD_DBG],[test x"$DBGSTATUS" = xyes] )
AM_CONDITIONAL([BUILD_MPI_DBG],[test x"$DBGSTATUS" = xyes -a x"$MPISTATUS" = xyes] )

TNL_INCLUDE_DIR="$prefix/include/tnl-$VERSION"
AC_SUBST(TNL_INCLUDE_DIR)

echo "Generating file libtnl-0.1.pc ..."

if [ test libtnl-0.1.pc ]
then
   rm -f libtnl-0.1.pc;
fi
echo "prefix=$prefix" > libtnl-0.1.pc
echo "exec_prefix=\${prefix}/bin" >> libtnl-0.1.pc
echo "libdir=\${prefix}/lib" >> libtnl-0.1.pc
echo "includedir=\${prefix}/include/tnl-$VERSION" >> libtnl-0.1.pc
echo >> libtnl-0.1.pc
echo "Name: TNL" >> libtnl-0.1.pc
echo "Description: Template Numerical Library." >> libtnl-0.1.pc
echo "Requires:" >> libtnl-0.1.pc
echo "Version: 0.1.0" >> libtnl-0.1.pc
echo "Libs: -L\${libdir} -ltnl-0.1" >> libtnl-0.1.pc
echo "Cflags: -I\${includedir} $BZIP2_CXXFLAGS" >> libtnl-0.1.pc

if test x"$enable_debug" == xyes
then
   echo "Generating file libtnl-dbg-0.1.pc ..."

   if [ test libtnl-dbg-0.1.pc ]
   then
      rm -f libtnl-dbg-0.1.pc;
   fi
   echo "prefix=$prefix" > libtnl-dbg-0.1.pc
   echo "exec_prefix=\${prefix}/bin" >> libtnl-dbg-0.1.pc
   echo "libdir=\${prefix}/lib" >> libtnl-dbg-0.1.pc
   echo "includedir=\${prefix}/include/tnl-$VERSION" >> libtnl-dbg-0.1.pc
   echo >> libtnl-dbg-0.1.pc
   echo "Name: TNL" >> libtnl-dbg-0.1.pc
   echo "Description: Template Numerical Library (with debuging information)." >> libtnl-dbg-0.1.pc
   echo "Requires:" >> libtnl-dbg-0.1.pc
   echo "Version: 0.1.0" >> libtnl-dbg-0.1.pc
   echo "Libs: -L\${libdir} -ltnl-dbg-0.1" >> libtnl-dbg-0.1.pc
   echo "Cflags: -I\${includedir} $BZIP2_CXXFLAGS" >> libtnl-dbg-0.1.pc
fi

if test x"$with_mpi" == xyes;
then
   if test x"$MPISTATUS" = xyes;
   then
      echo "Generating file libtnl-mpi-0.1.pc ..."
      if [ test libtnl-mpi-0.1.pc ]
      then
         rm -f libtnl-mpi-0.1.pc;
      fi
      echo "prefix=$prefix" > libtnl-mpi-0.1.pc
      echo "exec_prefix=\${prefix}/bin" >> libtnl-mpi-0.1.pc
      echo "libdir=\${prefix}/lib" >> libtnl-mpi-0.1.pc
      echo "includedir=\${prefix}/include/tnl-$VERSION" >> libtnl-mpi-0.1.pc
      echo >> libtnl-mpi-0.1.pc
      echo "Name: TNL" >> libtnl-mpi-0.1.pc
      echo "Description:  Template Numerical Library with MPI support." >> libtnl-mpi-0.1.pc
      echo "Requires:" >> libtnl-mpi-0.1.pc
      echo "Version: 0.1.0" >> libtnl-mpi-0.1.pc
      echo "Libs: $MPILDFLAGS  -L\${libdir}  -ltnl-mpi-0.1" >> libtnl-mpi-0.1.pc
      echo "Cflags: $MPICXXFLAGS -I\${includedir} $BZIP2_CXXFLAGS" >> libtnl-mpi-0.1.pc
      if test x"$enable_debug" == xyes
      then
         echo "Generating file libtnl-mpi-dbg-0.1.pc ..."

         if [ test libtnl-mpi-dbg-0.1.pc ]
         then
            rm -f libtnl-mpi-dbg-0.1.pc;
         fi
         echo "prefix=$prefix" > libtnl-mpi-dbg-0.1.pc
         echo "exec_prefix=\${prefix}/bin" >> libtnl-mpi-dbg-0.1.pc
         echo "libdir=\${prefix}/lib" >> libtnl-mpi-dbg-0.1.pc
         echo "includedir=\${prefix}/include/tnl-$VERSION" >> libtnl-mpi-dbg-0.1.pc
         echo >> libtnl-mpi-dbg-0.1.pc
         echo "Name: TNL" >> libtnl-mpi-dbg-0.1.pc
         echo "Description: Template Numerical Library with MPI support (and with debuging information). " >> libtnl-mpi-dbg-0.1.pc
         echo "Requires:" >> libtnl-mpi-dbg-0.1.pc
         echo "Version: 0.1.0" >> libtnl-mpi-dbg-0.1.pc
         echo "Libs: $MPILDFLAGS  -L\${libdir}  -ltnl-mpi-dbg-0.1" >> libtnl-mpi-dbg-0.1.pc
         echo "Cflags: $MPICXXFLAGS -I\${includedir} $BZIP2_CXXFLAGS" >> libtnl-mpi-dbg-0.1.pc
      fi #enable_debug == yes
   fi  #MPISTATUS == yes
fi #with_mpi == yes

echo "Generating sdf-restore-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > sdf-restore-def.h
echo "" >> sdf-restore-def.h
echo "#define CONFIG_DESCRIPTION_FILE \"$prefix/share/tnldist-0.2/sdf-restore.cfg.desc\"" >> sdf-restore-def.h
echo "" >> sdf-restore-def.h
echo "#define CPP_COMPILER_NAME \"$CXX\"" >> sdf-restore-def.h

echo "Generating mgrid-view-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > mgrid-view-def.h
echo "" >> mgrid-view-def.h
echo "#define MGRID_VIEW_CONFIG_DESCRIPTION_FILE \"$prefix/share/tnl-tools/mgrid-view.cfg.desc\"" >> mgrid-view-def.h

echo "Generating tnlcurve2gnuplot-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > tnlcurve2gnuplot-def.h
echo "" >> tnlcurve2gnuplot-def.h
echo "#define CONFIG_DESCRIPTION_FILE \"$prefix/share/tnl-tools/tnlcurve2gnuplot.cfg.desc\"" >> tnlcurve2gnuplot-def.h

echo "Generating mdiff-err-norms-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > mdiff-err-norms-def.h
echo "" >> mdiff-err-norms-def.h
echo "#define CONFIG_DESCRIPTION_FILE \"$prefix/share/tnl-tools/mdiff-err-norms.cfg.desc\"" >> mdiff-err-norms-def.h

echo "Generating tnl-matrix-convert-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > tnl-matrix-convert-def.h
echo "" >> tnl-matrix-convert-def.h
echo "#define CONFIG_DESCRIPTION_FILE \"$prefix/share/tnl-tools/tnl-matrix-convert.cfg.desc\"" >> tnl-matrix-convert-def.h

echo "Generating tnl-env-variables"
PWD=`pwd`
TNL_SOURCE_DIR=`echo $PWD | sed 's/\/build//'`
echo "TNL_SOURCE_DIR=$TNL_SOURCE_DIR" > tnl-env-variables

echo "Generating matrix-formats-test-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > matrix-formats-test-def.h
echo "" >> matrix-formats-test-def.h
echo "#define CONFIG_DESCRIPTION_FILE \"$TNL_SOURCE_DIR/tests/matrix-formats-test.cfg.desc\"" >> matrix-formats-test-def.h

echo "Generating sparse-matrix-benchmark-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > sparse-matrix-benchmark-def.h
echo "" >> sparse-matrix-benchmark-def.h
echo "#define CONFIG_DESCRIPTION_FILE \"$TNL_SOURCE_DIR/tests/sparse-matrix-benchmark.cfg.desc\"" >> sparse-matrix-benchmark-def.h

echo "Generating matrix-solvers-benchmark-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > matrix-solvers-benchmark-def.h
echo "" >> matrix-solvers-benchmark-def.h
echo "#define CONFIG_DESCRIPTION_FILE \"$TNL_SOURCE_DIR/tests/matrix-solvers-benchmark.cfg.desc\"" >> matrix-solvers-benchmark-def.h





CXXFLAGS="$CXXFLAGS  $BZIP2_CXXFLAGS $OPENMPFLAGS $CUDA_CXXFLAGS -I../src/debug -I../src/core -I../src/diff -I../../src -I../../../src"
CXXFLAGS="$CXXFLAGS -I../../src -I../../src/debug -I../src/debug"
CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS"
LDFLAGS="$LDFLAGS $CUDA_LDFLAGS -L/usr/lib -L/usr/local/lib -lamd"
OPTIMISECXXFLAGS="-O3"

AC_SUBST(OPTIMISECXXFLAGS)



AC_OUTPUT(Makefile \
          src/core/Makefile \
          src/core/low-level/Makefile \
          src/config/Makefile \
          src/debug/Makefile \
          src/diff/Makefile \
          src/matrix/Makefile \
          src/mesh/Makefile \
          src/solver/Makefile \
          src/Makefile \
          tools/Makefile \
          tools/share/Makefile \
          tools/src/Makefile \
          tests/Makefile \
          tests/unit-tests/Makefile)



echo
echo "**************************************************"
echo "*                                                *"
echo "*                TNL Library                     *"
echo "*                                                *"
echo "**************************************************"
echo "*"
echo "* Compiler:" $CXX
echo "* Preprocessor:" $CPP
echo "* Compiler flags:" $CXXFLAGS
echo "* Preprocessor flags:" $CPPFLAGS
echo "* Linker flags: " $LDFLAGS $LIBS
echo "* Debug flags: " $DBGCXXFLAGS
echo "* Optimising flags: " $OPTIMISECXXFLAGS
echo "* MPI compiler flags:" $MPICXXFLAGS
echo "* MPI linker flags:" $MPILDFLAGS
echo "* CUDA flags:" $CUDA_CXXFLAGS
echo "* Testing CUSP flags:" $CUSP_CXXFLAGS
 


if eval test "x$enable_debug = xyes"; then
   echo "* Debug mode: yes"
fi
if eval test "x$with_gprof = xyes"; then
   echo "* We use PROFILE mode."
fi
if eval test "x$with_dmalloc = xfull"; then
   echo "* Linking with dmalloc."
fi
if eval test "x$with_efence = xyes"; then
   echo "* Linking with Electric Fence."
fi
if eval test "x$GNUPLOT = xno"; then
   echo "* WARNING! Gnuplot was not found."
fi
echo "*"
echo "**************************************************"
echo "*  Now enter 'make' followed by 'make install'.  *"
echo "**************************************************"


echo "#ifndef TNL_CONFIGH " > tnl-config.h
echo "#define TNL_CONFIGH " >> tnl-config.h
cat config.h >> tnl-config.h
echo "#endif" >> tnl-config.h

cp -f tools/src/tnl-err-norms.cpp tools/src/tnl-err-norms-cuda.cu
cp -f ../tools/src/tnl-err-norms.cpp tools/src/tnl-err-norms-cuda.cu

cp -f tools/src/tnl-grid-view.cpp tools/src/tnl-grid-view-cuda.cu
cp -f ../tools/src/tnl-grid-view.cpp tools/src/tnl-grid-view-cuda.cu

cp -f tests/unit-tests/tnl-unit-tests.cpp tests/unit-tests/tnl-unit-tests-cuda.cu
cp -f ../tests/unit-tests/tnl-unit-tests.cpp tests/unit-tests/tnl-unit-tests-cuda.cu

cp -f tests/matrix-formats-test.cpp tests/matrix-formats-test-cuda.cu
cp -f ../tests/matrix-formats-test.cpp tests/matrix-formats-test-cuda.cu

cp -f tests/sparse-matrix-benchmark.cpp tests/sparse-matrix-benchmark-cuda.cu
cp -f ../tests/sparse-matrix-benchmark.cpp tests/sparse-matrix-benchmark-cuda.cu

cp -f tests/matrix-solvers-benchmark.cpp tests/matrix-solvers-benchmark-cuda.cu
cp -f ../tests/matrix-solvers-benchmark.cpp tests/matrix-solvers-benchmark-cuda.cu

cp -f tests/tnl-benchmarks.cpp tests/tnl-benchmarks-cuda.cu
cp -f ../tests/tnl-benchmarks.cpp tests/tnl-benchmarks-cuda.cu

