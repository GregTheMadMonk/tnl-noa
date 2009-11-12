dnl Process this file with autoconf to produce a configure script
AC_INIT(tnl,0.1)
AC_CONFIG_MACRO_DIR([m4])

dnl automake init
AM_INIT_AUTOMAKE( libtnl, 0.1 )
AM_CONFIG_HEADER( config.h )
AM_PROG_LIBTOOL

dnl -----------------------------------------
dnl ----------- check for icpc --------------
dnl -----------------------------------------

if test x"$CXX" = xicpc; then
   CXXFLAGS="$CXXFLAGS -DHAVE_ICPC"
fi

dnl -----------------------------------------
dnl ----------- check for debug--------------
dnl -----------------------------------------
AC_ARG_ENABLE(debug,[  --enable-debug=[no/yes]	turn on debugging [default=no]] )
if test x"$enable_debug" = xyes; then
   CXXFLAGS="$CXXFLAGS -O0 -DDEBUG -g3 -Wall -W -ansi -Wno-unused"
else
   CXXFLAGS="$CXXFLAGS -O3"
fi

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


AC_PROG_CXX
AC_PROG_AWK

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
   AC_LANG_SAVE
   AC_LANG_CPLUSPLUS
   CXXFLAGS="$CXXFLAGS $MPICXXFLAGS"
   AC_CHECK_HEADERS(mpi.h, MPICXXFLAGS="$MPICXXFLAGS -DHAVE_MPI")
   AC_LANG_RESTORE
   CXXFLAGS="$STORE_CXXFLAGS"
   AM_CONDITIONAL(BUILDMPI,test x"$MPISTATUS" = xyes )
   AC_SUBST(MPICXXFLAGS)
   AC_SUBST(MPILDFLAGS)
else
   AM_CONDITIONAL(BUILDMPI,test xno = xyes )
fi

dnl -----------------------------------------
dnl ---------- check for OpenMP -------------
dnl -----------------------------------------

AC_ARG_WITH(openmp, 
            AS_HELP_STRING([--with-openmp=[yes/no],compile with OpenMP support[default=no]]),
            with_openmp=$withval,
            with_openmp=no )
if test x"$with_openmp" = xyes;
then
   AC_CHECK_HEADER(omp.h, OPENMPFLAGS="-openmp -DHAVE_OPENMP" )
fi

CXXFLAGS="$CXXFLAGS -I../core -I../debug -I../diff -I../dist"

dnl cd src/mpi
dnl rm -rf *.cpp *.h *.ih *.cc
dnl ln -s ../*.cpp ../*.h ../*.ih ../*.cc .
dnl cd ../..

echo "Generating file libtnl-0.1.pc ..."

if [ test libtnl-0.1.pc ]
then
   rm -f libtnl-0.1.pc;
fi
echo "prefix=$prefix" > libtnl-0.1.pc
echo "exec_prefix=\${prefix}/bin" >> libtnl-0.1.pc
echo "libdir=\${prefix}/lib" >> libtnl-0.1.pc
echo "includedir=\${prefix}/include" >> libtnl-0.1.pc
echo >> libtnl-0.1.pc
echo "Name: TNL" >> libtnl-0.1.pc
echo "Description: Template Numerical Library." >> libtnl-0.1.pc
echo "Requires:" >> libtnl-0.1.pc
echo "Version: 0.1.0" >> libtnl-0.1.pc
echo "Libs: -L\${libdir} -ltnldebug-0.1 -ltnlcore-0.1 -ltnldiff-0.1 -ltnldist-0.1 -ltnlmatrix-0.1 " >> libtnl-0.1.pc
echo "Cflags: -I\${includedir}/tnldebug-0.1 -I\${includedir}/tnlcore-0.1 -I\${includedir}/tnldiff-0.1 -I\${includedir}/tnldist-0.1 -I\${includedir}/tnlmatrix-0.1 " >> libtnl-0.1.pc

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
      echo "includedir=\${prefix}/include" >> libtnl-mpi-0.1.pc
      echo >> libtnl-mpi-0.1.pc
      echo "Name: TNL" >> libtnl-mpi-0.1.pc
      echo "Description: The core of the system for numerical computation with MPI support." >> libtnl-mpi-0.1.pc
      echo "Requires:" >> libtnl-mpi-0.1.pc
      echo "Version: 0.1.0" >> libtnl-mpi-0.1.pc
      echo "Libs: $MPILDFLAGS  -L\${libdir}  -ltnldebug-mpi-0.1 -ltnlcore-mpi-0.1 -ltnldiff-mpi-0.1 -ltnldist-mpi-0.1 -ltnlmatrix-mpi-0.1" >> libtnl-mpi-0.1.pc
      echo "Cflags: $MPICXXFLAGS -I\${includedir}  -I\${includedir}/tnldebug-0.1 -I\${includedir}/tnlcore-0.1 -I\${includedir}/tnldiff-0.1 -I\${includedir}/tnldist-0.1 -I\${includedir}/tnlmatrix-0.1 " >> libtnl-mpi-0.1.pc
   fi
fi

echo "Generating sdf-restore-def.h"

echo "// This file is generated by the ./configure script. Do not change it!" > sdf-restore-def.h
echo "" >> sdf-restore-def.h
echo "#define CONFIG_DESCRIPTION_FILE \"$prefix/share/tnldist-0.2/sdf-restore.cfg.desc\"" >> sdf-restore-def.h
echo "" >> sdf-restore-def.h
echo "#define CPP_COMPILER_NAME \"$CXX\"" >> sdf-restore-def.h


CXXFLAGS="$CXXFLAGS $OPENMPFLAGS -I../src/debug -I../src/core -I../src/diff"

AC_OUTPUT(Makefile
          src/Makefile
          src/core/Makefile
          src/debug/Makefile
          src/diff/Makefile
          src/dist/Makefile
          src/matrix/Makefile
          )

echo
echo "**************************************************"
echo "*                                                *"
echo "*                TNL Library                     *"
echo "*                                                *"
echo "**************************************************"
echo "*"
echo "* Compiler:" $CXX
echo "* Compiler flags:" $CXXFLAGS
echo "* Linker flags: " $LDFLAGS
echo "* MPI compiler flags:" $MPICXXFLAGS
echo "* MPI linker flags:" $MPILDFLAGS


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
