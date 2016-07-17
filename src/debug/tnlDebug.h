/***************************************************************************
                          tnlDebug.h  -  description
                             -------------------
    begin                : 2004/09/05
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef __tnlDebug_h__
#define __tnlDebug_h__


#ifdef DEBUG
#include <assert.h>
#include <iostream>
#include <stdio.h>

#include <string>

#ifdef HAVE_MPI
   #include <mpi.h>
extern int __tnldbg_mpi_i_proc;
#endif

#define dbgInit( file_name ) \
   tnlInitDebug( file_name );

#define dbgFunctionName( _class, _func )     \
   const char* _tnldbg_debug_class_name = _class; \
   const char* _tnldbg_debug_func_name =  _func;

#ifdef HAVE_MPI // MPI definitions

#define dbgMPIBarrier                                                          \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,                           \
                       _tnldbg_debug_func_name ) )                             \
    MPI_Barrier( MPI_COMM_WORLD );

#define dbgCout( args )                                                        \
   MPI_Comm_rank( MPI_COMM_WORLD, &__tnldbg_mpi_i_proc );                      \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,                           \
                       _tnldbg_debug_func_name ) )                             \
      std :: cout << "#TNLDBG# MPI proc. " << __tnldbg_mpi_i_proc << " | "     \
      << _tnldbg_debug_class_name << " :: "                                    \
      << _tnldbg_debug_func_name << " @ "                                      \
      << __LINE__ << " | " << args << std :: endl << std :: flush;

#define dbgCoutLine( args )                                                    \
   MPI_Comm_rank( MPI_COMM_WORLD, &__tnldbg_mpi_i_proc );                      \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,                           \
                       _tnldbg_debug_func_name ) )                             \
      std :: cout << "#TNLDBG# MPI proc. " << __tnldbg_mpi_i_proc << " | "     \
      << _tnldbg_debug_class_name << " :: "                                    \
      << _tnldbg_debug_func_name << " @ "                                      \
      << __LINE__ << " | " << args << "            \r" << std :: flush;


#define dbgExpr( expr )                                                        \
   MPI_Comm_rank( MPI_COMM_WORLD, &__tnldbg_mpi_i_proc );                      \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,                           \
                           _tnldbg_debug_func_name ) )                         \
      std :: cout << "#TNLDBG# MPI proc. " << __tnldbg_mpi_i_proc << " | "     \
      << _tnldbg_debug_class_name << " :: "                                    \
      << _tnldbg_debug_func_name << " @ "                                      \
      << __LINE__ << " | " << #expr << " -> "                                  \
      << expr << std :: endl << std :: flush;

#define dbgExprLine( expr )                                                        \
   MPI_Comm_rank( MPI_COMM_WORLD, &__tnldbg_mpi_i_proc );                      \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,                           \
                           _tnldbg_debug_func_name ) )                         \
      std :: cout << "#TNLDBG# MPI proc. " << __tnldbg_mpi_i_proc << " | "     \
      << _tnldbg_debug_class_name << " :: "                                    \
      << _tnldbg_debug_func_name << " @ "                                      \
      << __LINE__ << " | " << #expr << " -> "                                  \
      << expr << "              \r" << std :: flush;


#define dbgCondExpr( condition, expr )                                         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,                           \
                       _tnldbg_debug_func_name ) )                             \
      if( condition )                                                          \
         std :: cout << "#TNLDBG# MPI proc. " << __tnldbg_mpi_i_proc << " | "         \
         << _tnldbg_debug_class_name << " :: "                                 \
         << _tnldbg_debug_func_name << " @ "                                   \
         << __LINE__ << " | "                                                  \
      << #expr << " -> " << expr << std :: flush << std :: endl;

#else   // now non-MPI definitions

#define dbgCout( args )                                                        \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,                           \
                       _tnldbg_debug_func_name ) )                             \
      std :: cout << "#TNLDBG# " << _tnldbg_debug_class_name << " :: "         \
      << _tnldbg_debug_func_name << " @ "                                      \
      << __LINE__ << " | " << args << std :: flush << std :: endl;

#define dbgCoutLine( args )                                                    \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,                           \
                       _tnldbg_debug_func_name ) )                             \
      std :: cout << "#TNLDBG# " << _tnldbg_debug_class_name << " :: "         \
      << _tnldbg_debug_func_name << " @ "                                      \
      << __LINE__ << " | " << args << "             \r" << std :: flush;


#define dbgPrintf( expr )                                    \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,         \
                           _tnldbg_debug_func_name ) )       \
   {                                                         \
      printf( "#TNLDBG# %s :: %s @ %d |" ,                           \
      _tnldbg_debug_class_name,  _tnldbg_debug_func_name,    \
      __LINE__);                                             \
      printf( #expr );                                       \
      printf( "\n" );                                        \
   }

#define dbgExpr( expr )                                      \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,         \
                           _tnldbg_debug_func_name ) )       \
      std :: cout << "#TNLDBG# " << _tnldbg_debug_class_name << " :: "     \
      << _tnldbg_debug_func_name << " @ "                    \
      << __LINE__ << " | "                                   \
      << #expr << " -> " << expr << std :: flush << std :: endl;

#define dbgExprLine( expr )                                      \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,         \
                           _tnldbg_debug_func_name ) )       \
      std :: cout << "#TNLDBG# " << _tnldbg_debug_class_name << " :: "     \
      << _tnldbg_debug_func_name << " @ "                    \
      << __LINE__ << " | "                                   \
      << #expr << " -> " << expr << "                 \r" << std :: flush;


#define dbgCall( expr )                         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                           _tnldbg_debug_func_name ) )  \
   {                                             \
      std :: cout << "#TNLDBG#" << _tnldbg_debug_class_name << " :: "    \
      << _tnldbg_debug_func_name << " @ "           \
      << __LINE__ << " | CALL:  "                \
      << #expr << " -> " << std :: flush << std :: endl;       \
      expr;                                      \
   }

#define dbgWait                                     \
   if( _tnldbg_interactive_func( _tnldbg_debug_class_name, \
                       _tnldbg_debug_func_name ) )      \
   {                                                 \
      std :: cout << "#TNLDBG#" <<  _tnldbg_debug_class_name << " :: "        \
      << _tnldbg_debug_func_name << " @ "               \
      << __LINE__ << " | Press any key...  "         \
      << std :: flush << std :: endl;                              \
      getchar();                                     \
   }

#define dbgCondCout( condition, args )         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                       _tnldbg_debug_func_name ) )  \
      if( condition )                            \
         std :: cout << "#TNLDBG#" <<  _tnldbg_debug_class_name << " :: " \
         << _tnldbg_debug_func_name << " @ "        \
         << __LINE__ << " | " << args << std :: flush << std :: endl;

#define dbgCondPrintf( condition, expr )             \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,         \
                       _tnldbg_debug_func_name ) )        \
      if( condition )                                  \
      {                                                \
         printf( "#TNLDBG# %s :: %s @ %d |" ,                  \
         _tnldbg_debug_class_name,  _tnldbg_debug_func_name, \
         __LINE__);                                    \
         printf( #expr );                              \
         printf( "\n" );                               \
      }

#define dbgCondExpr( condition, expr )         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                       _tnldbg_debug_func_name ) )  \
      if( condition )                            \
         std :: cout << "#TNLDBG#" << _tnldbg_debug_class_name << " :: " \
         << _tnldbg_debug_func_name << " @ "        \
         << __LINE__ << " | "                    \
      << #expr << " -> " << expr << std :: flush << std :: endl;

#define dbgCondCall( condition, expr )         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                       _tnldbg_debug_func_name ) )  \
      if( condition )                            \
      {                                          \
         std :: cout << "#TNLDBG#" <<  _tnldbg_debug_class_name << " :: " \
         << _tnldbg_debug_func_name << " @ "        \
         << __LINE__ << " | CALL:  "             \
         << #expr << " -> " << std :: flush << std :: endl;    \
         expr;                                   \
      }

#define dbgCondWait( condition )                     \
   if( _tnldbg_interactive_func( _tnldbg_debug_class_name,   \
                              _tnldbg_debug_func_name ) ) \
      if( condition )                                  \
      {                                                \
         std :: cout << "#TNLDBG#" <<  _tnldbg_debug_class_name << " :: "       \
         << _tnldbg_debug_func_name << " @ "              \
         << __LINE__ << " | Press any key...  "        \
         << std :: flush << std :: endl;                             \
         getchar();                                    \
      }

#define dbgCoutArray( array, size )                                           \
   if( _tnldbg_interactive_func( _tnldbg_debug_class_name,                            \
                              _tnldbg_debug_func_name ) )                          \
   {                                                                            \
      std :: cout << "#TNLDBG#" << _tnldbg_debug_class_name << " :: "                                   \
          << _tnldbg_debug_func_name << " @ "                                      \
          << __LINE__ << " | "                                                  \
          << #array << " = [ ";                                                 \
      for( int _tnldbg_index_ii = 0; _tnldbg_index_ii < size - 1 ; _tnldbg_index_ii ++ ) \
         std :: cout << array[ _tnldbg_index_ii ] << ", ";                                \
      std :: cout << array[ size - 1 ] << " ]" << std :: flush << std :: endl;                       \
   }

#define dbgCoutMatrixCW( matrix, n1, n2, wd )                                     \
   if( _tnldbg_interactive_func( _tnldbg_debug_class_name,                                 \
                              _tnldbg_debug_func_name ) )                               \
   {                                                                                 \
      std :: cout "#TNLDBG#" <<  << _tnldbg_debug_class_name << " :: "                                        \
          << _tnldbg_debug_func_name << " @ "                                           \
          << __LINE__ << " | "                                                       \
          << #matrix << " = [ " << std :: endl;                                             \
      for( int _tnldbg_index_ii = 0; _tnldbg_index_ii < ( n1 ) ; _tnldbg_index_ii ++ ) {      \
         for( int _tnldbg_index_jj = 0; _tnldbg_index_jj < ( n2 ) - 1 ; _tnldbg_index_jj ++ ) \
            std :: cout << setw( wd )                                                       \
                 << matrix[ _tnldbg_index_jj * ( n1 ) + _tnldbg_index_ii ] << ", ";        \
            std :: cout << setw( wd )                                                       \
                 << matrix[ ( ( n2 ) - 1 ) * ( n1 ) + _tnldbg_index_ii ] << std :: endl;}      \
       std :: cout << " ]" << std :: flush << std :: endl;                                                 \
   }

#define dbgMPIBarrier

#endif

bool tnlInitDebug( const char* file_name,
                   const char* program_name = 0 );
bool _tnldbg_debug_func( const char* group_name,
                      const char* function_name );
bool _tnldbg_interactive_func( const char* group_name,
                            const char* function_name );

#else
   #define dbgInit( file_name )
   #define dbgFunctionName( _class, _func )
   #define dbgCout( args )
   #define dbgCoutLine( args )
   #define dbgPrintf( expr )
   #define dbgExpr( expr )
   #define dbgExprLine( expr )
   #define dbgCall( expr )
   #define dbgWait
   #define dbgCondCout( condition, args )
   #define dbgCondPrintf( condition, expr )
   #define dbgCondExpr( condition, expr )
   #define dbgCondCall( condition, expr )
   #define dbgCondWait( condition )
   #define dbgCondArray( array, size )
   #define dbgCondMatrixCW( matrix, n1, n2, wd )
   #define dbgMPIBarrier
#endif


#endif
