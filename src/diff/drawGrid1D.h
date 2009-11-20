/***************************************************************************
                          drawGrid1D.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef drawGrid1DH
#define drawGrid1DH

#include <iostream>
#include <fstream>
#include <iomanip>
#include <diff/tnlGrid1D.h>
#include <diff/mGridSystem1D.h>
#include <core/mfuncs.h>
#include <core/compress-file.h>

using namespace std;

template< class T > bool Draw( const tnlGrid1D< T >& u,
                               ostream& str,
                               const char* format,
                               const long int i_step = 1 )
{
   if( ! format )
   {
      cerr << "No format given for drawing 1D grid. " << endl;
      return false;
   }
   long int i;
   const long int x_size = u. GetXSize();
   const double& ax = u. GetAx();
   const double& hx = u. GetHx();
   if( strcmp( format, "gnuplot" ) == 0 )
   {
      for( i = 0; i < x_size; i += i_step )
            str << setprecision( 12 ) << ax + i * hx << " " << u( i ) << endl;
      return true;           
   }
   if( strncmp( format, "bin", 3 ) == 0 )
   {
      if( ! u. Save( str ) ) return false;
      return true;
   }
   cerr << "Unknown format '" << format << "' for drawing a grid 1D." << endl;
   return false;
};

template< class T, int SYSTEM_SIZE, typename SYSTEM_INDEX > bool Draw( const mGridSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX >& u,
                                                                       ostream& str,
                                                                       const char* format,
                                                                       const long int i_step = 1 )
{
   if( ! format )
   {
      cerr << "No format given for drawing 1D grid system. " << endl;
      return false;
   }
   long int i, j;
   const long int x_size = u. GetXSize();
   const double& ax = u. GetAx();
   const double& hx = u. GetHx();
   if( strcmp( format, "gnuplot" ) == 0 )
   {
      for( i = 0; i < x_size; i += i_step )
      {
         str << setprecision( 12 ) << ax + i * hx;
         for( j = 0; j < SYSTEM_SIZE; j ++ )
            str << " " << u( ( SYSTEM_INDEX ) j, i );
         str << endl;
      }
      return true;           
   }
   if( strncmp( format, "bin", 3 ) == 0 )
   {
      if( ! u. Save( str ) ) return false;
      return true;
   }
   cerr << "Unknown format '" << format << "' for drawing a grid system 1D." << endl;
   return false;
};

template< class T > bool Draw( const tnlGrid1D< T >& u,
                               const char* file_name,
                               const char* format,
                               const long int i_step = 1 )
{
   fstream file;
   if( strncmp( format, "bin",3 ) == 0 )
      file. open( file_name, ios :: out | ios :: binary );
   else file. open( file_name, ios :: out );
   if( ! file )
   {
      cerr << "Sorry I can not open the file " << file_name << endl;
      return false;
   }
   bool result = Draw( u, file, format, i_step );
   file. close();
   if( ! result ) return false;
   int len = strlen( format );
   if( strcmp( format + Max( 0, len - 3 ), "-gz" ) == 0 && 
       ! CompressFile( file_name, "gz" ) )
      return false;
   if( strcmp( format + Max( 0, len - 4 ), "-bz2" ) == 0  &&
       ! CompressFile( file_name, "bz2" ) )
         return false;
   return true;
};

template< class T, int SYSTEM_SIZE, typename SYSTEM_INDEX > bool Draw( const mGridSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX >& u,
                                                                       const char* file_name,
                                                                       const char* format,
                                                                       const long int i_step = 1 )
{
   fstream file;
   if( strncmp( format, "bin",3 ) == 0 )
      file. open( file_name, ios :: out | ios :: binary );
   else file. open( file_name, ios :: out );
   if( ! file )
   {
      cerr << "Sorry I can not open the file " << file_name << endl;
      return false;
   }
   bool result = Draw( u, file, format, i_step );
   file. close();
   if( ! result )
   {
      cerr << "Sorry I could not write to the file " << file_name << endl;
      return false;
   }
   int len = strlen( format );
   if( strcmp( format + Max( 0, len - 3 ), "-gz" ) == 0 && 
       ! CompressFile( file_name, "gz" ) )
      return false;
   if( strcmp( format + Max( 0, len - 4 ), "-bz2" ) == 0  &&
       ! CompressFile( file_name, "bz2" ) )
         return false;
   return true;
};


#endif
