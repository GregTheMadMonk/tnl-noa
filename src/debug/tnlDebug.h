/***************************************************************************
                          tnlDebug.h  -  description
                             -------------------
    begin                : 2004/09/05
    copyright            : (C) 2004 by Tom� Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef __tnlDebug_h__
#define __tnlDebug_h__


#ifdef DEBUG 
#include <assert.h>
#include <iostream.h>
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

#define dbgMPIBarrier                          \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                       _tnldbg_debug_func_name ) )  \
    MPI_Barrier( MPI_COMM_WORLD );

#define dbgCout( args )                                       \
   MPI_Comm_rank( MPI_COMM_WORLD, &__tnldbg_mpi_i_proc );     \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,          \
                       _tnldbg_debug_func_name ) )            \
      cout << "# MPI proc. " << __tnldbg_mpi_i_proc << " | "  \
      << _tnldbg_debug_class_name << " :: "                   \
      << _tnldbg_debug_func_name << " @ "                     \
      << __LINE__ << " | " << args << endl << flush;

#define dbgExpr( expr )                                       \
   MPI_Comm_rank( MPI_COMM_WORLD, &__tnldbg_mpi_i_proc );     \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,          \
                           _tnldbg_debug_func_name ) )        \
      cout << "# MPI proc. " << __tnldbg_mpi_i_proc << " | "  \
      << _tnldbg_debug_class_name << " :: "                   \
      << _tnldbg_debug_func_name << " @ "                     \
      << __LINE__ << " | " << #expr << " -> "                 \
      << expr << endl << flush;

#define dbgCondExpr( condition, expr )         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                       _tnldbg_debug_func_name ) )  \
      if( condition )                            \
         cout << "# MPI proc. " << __tnldbg_mpi_i_proc << " | "  \
         << _tnldbg_debug_class_name << " :: " \
         << _tnldbg_debug_func_name << " @ "        \
         << __LINE__ << " | "                    \
      << #expr << " -> " << expr << flush << endl;

#else   // now non-MPI definitions

#define dbgCout( args )                                      \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,         \
                       _tnldbg_debug_func_name ) )           \
      cout << "# " << _tnldbg_debug_class_name << " :: "     \
      << _tnldbg_debug_func_name << " @ "                    \
      << __LINE__ << " | " << args << flush << endl;

#define dbgPrintf( expr )                                    \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,         \
                           _tnldbg_debug_func_name ) )       \
   {                                                         \
      printf( " %s :: %s @ %d |" ,                           \
      _tnldbg_debug_class_name,  _tnldbg_debug_func_name,    \
      __LINE__);                                             \
      printf( #expr );                                       \
      printf( "\n" );                                        \
   }         

#define dbgExpr( expr )                                      \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,         \
                           _tnldbg_debug_func_name ) )       \
      cout << "# " << _tnldbg_debug_class_name << " :: "     \
      << _tnldbg_debug_func_name << " @ "                    \
      << __LINE__ << " | "                                   \
      << #expr << " -> " << expr << flush << endl;

#define dbgCall( expr )                         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                           _tnldbg_debug_func_name ) )  \
   {                                             \
      cout << _tnldbg_debug_class_name << " :: "    \
      << _tnldbg_debug_func_name << " @ "           \
      << __LINE__ << " | CALL:  "                \
      << #expr << " -> " << flush << endl;       \
      expr;                                      \
   }         

#define dbgWait                                     \
   if( _tnldbg_interactive_func( _tnldbg_debug_class_name, \
                       _tnldbg_debug_func_name ) )      \
   {                                                 \
      cout << _tnldbg_debug_class_name << " :: "        \
      << _tnldbg_debug_func_name << " @ "               \
      << __LINE__ << " | Press any key...  "         \
      << flush << endl;                              \
      getchar();                                     \
   }

#define dbgCondCout( condition, args )         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                       _tnldbg_debug_func_name ) )  \
      if( condition )                            \
         cout << _tnldbg_debug_class_name << " :: " \
         << _tnldbg_debug_func_name << " @ "        \
         << __LINE__ << " | " << args << flush << endl;

#define dbgCondPrintf( condition, expr )             \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,         \
                       _tnldbg_debug_func_name ) )        \
      if( condition )                                  \
      {                                                \
         printf( " %s :: %s @ %d |" ,                  \
         _tnldbg_debug_class_name,  _tnldbg_debug_func_name, \
         __LINE__);                                    \
         printf( #expr );                              \
         printf( "\n" );                               \
      }         

#define dbgCondExpr( condition, expr )         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                       _tnldbg_debug_func_name ) )  \
      if( condition )                            \
         cout << _tnldbg_debug_class_name << " :: " \
         << _tnldbg_debug_func_name << " @ "        \
         << __LINE__ << " | "                    \
      << #expr << " -> " << expr << flush << endl;

#define dbgCondCall( condition, expr )         \
   if( _tnldbg_debug_func( _tnldbg_debug_class_name,   \
                       _tnldbg_debug_func_name ) )  \
      if( condition )                            \
      {                                          \
         cout << _tnldbg_debug_class_name << " :: " \
         << _tnldbg_debug_func_name << " @ "        \
         << __LINE__ << " | CALL:  "             \
         << #expr << " -> " << flush << endl;    \
         expr;                                   \
      }         

#define dbgCondWait( condition )                     \
   if( _tnldbg_interactive_func( _tnldbg_debug_class_name,   \
                              _tnldbg_debug_func_name ) ) \
      if( condition )                                  \
      {                                                \
         cout << _tnldbg_debug_class_name << " :: "       \
         << _tnldbg_debug_func_name << " @ "              \
         << __LINE__ << " | Press any key...  "        \
         << flush << endl;                             \
         getchar();                                    \
      }

#define dbgCoutArray( array, size )                                           \
   if( _tnldbg_interactive_func( _tnldbg_debug_class_name,                            \
                              _tnldbg_debug_func_name ) )                          \
   {                                                                            \
      cout << _tnldbg_debug_class_name << " :: "                                   \
          << _tnldbg_debug_func_name << " @ "                                      \
          << __LINE__ << " | "                                                  \
          << #array << " = [ ";                                                 \
      for( int _tnldbg_index_ii = 0; _tnldbg_index_ii < size - 1 ; _tnldbg_index_ii ++ ) \
         cout << array[ _tnldbg_index_ii ] << ", ";                                \
      cout << array[ size - 1 ] << " ]" << flush << endl;                       \
   }

#define dbgCoutMatrixCW( matrix, n1, n2, wd )                                     \
   if( _tnldbg_interactive_func( _tnldbg_debug_class_name,                                 \
                              _tnldbg_debug_func_name ) )                               \
   {                                                                                 \
      cout << _tnldbg_debug_class_name << " :: "                                        \
          << _tnldbg_debug_func_name << " @ "                                           \
          << __LINE__ << " | "                                                       \
          << #matrix << " = [ " << endl;                                             \
      for( int _tnldbg_index_ii = 0; _tnldbg_index_ii < ( n1 ) ; _tnldbg_index_ii ++ ) {      \
         for( int _tnldbg_index_jj = 0; _tnldbg_index_jj < ( n2 ) - 1 ; _tnldbg_index_jj ++ ) \
            cout << setw( wd )                                                       \
                 << matrix[ _tnldbg_index_jj * ( n1 ) + _tnldbg_index_ii ] << ", ";        \
            cout << setw( wd )                                                       \
                 << matrix[ ( ( n2 ) - 1 ) * ( n1 ) + _tnldbg_index_ii ] << endl;}      \
      cout << " ]" << flush << endl;                                                 \
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
   #define dbgPrintf( expr )
   #define dbgExpr( expr )
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
