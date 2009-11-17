/***************************************************************************
                          drawGrid2D.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#ifndef drawGrid2DH
#define drawGrid2DH

#include <ostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <diff/mGrid2D.h>
#include <core/compress-file.h>
#include <core/mfuncs.h>

template< class T > bool Draw( const mGrid2D< T >& u,
                               ostream& str,
                               const char* format,
                               const long int i_step = 1,
                               const long int j_step = 1 )
{
   if( ! format )
   {
      cerr << "No format given for drawing 2D grid. " << endl;
      return false;
   }
   long int i, j;
   const long int x_size = u. GetXSize();
   const long int y_size = u. GetYSize();
   const double& ax = u. GetAx();
   const double& ay = u. GetAy();
   const double& hx = u. GetHx();
   const double& hy = u. GetHy();
   if( strcmp( format, "gnuplot" ) == 0 )
   {
      for( i = 0; i < x_size; i += i_step )
      {
         for( j = 0; j < y_size; j += j_step )
            str << setprecision( 12 ) << ax + i * hx << " " << ay + j * hy << " " << u( i, j ) << endl;
         str << endl;
      }
      return true;           
   }
   if( strcmp( format, "vti" ) == 0 )
   {
      str << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
      str << "<ImageData WholeExtent=\"" 
           << 0 << " " << x_size - 1 << " " << 0 << " " << y_size - 1 
           << " 0 0\" Origin=\"0 0 0\" Spacing=\"" 
           << hx << " " << hy << " 0\">" << endl;
      str << "<Piece Extent=\"0 " << x_size - 1 << " 0 " << y_size - 1 <<" 0 0\">" << endl;
      str << "<PointData Scalars=\"order_parameter\">" << endl;
      str << "<DataArray Name=\"order_parameter\" type=\"Float32\" format=\"ascii\">" << endl;
      str. flags( ios_base::scientific );
      long int i, j;
      for( j = 0; j <= y_size - j_step; j += j_step )
         for( i = 0; i <= x_size - i_step; i += i_step )
              str << u( i, j ) << " ";
      str << endl;
      str << "</DataArray>" << endl;
      str << "</PointData>" << endl;
      str << "</Piece>" << endl;
      str << "</ImageData>" << endl;
      str << "</VTKFile>" << endl;
      return true;
   }
   if( strncmp( format, "bin", 3 ) == 0 )
   {
      if( ! u. Save( str ) ) return false;
      return true;
   }
   cerr << "Unknown format '" << format << "' for drawing a grid 2D." << endl;
   return false;
};

template< class T > bool Draw( const mGrid2D< T >& u,
                               const char* file_name,
                               const char* format,
                               const long int i_step = 1,
                               const long int j_step = 1 )
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
   bool result = Draw( u, file, format, i_step, j_step );
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

template< class T > bool Read( mGrid2D< T >& u,
                               const char* input_file )
{
   int strln = strlen( input_file );
   mString uncompressed_file_name( input_file );
   if( strcmp( input_file + strln - 3, ".gz" ) == 0 )
      if( ! UnCompressFile( input_file, "gz" ) )
      {
         cerr << "Unable to uncompress the file " << input_file << "." << endl;
         return false;
      }
      else uncompressed_file_name. SetString( input_file, 0, 3 );
   if( strcmp( input_file + strln - 4, ".bz2" ) == 0 )
      if( ! UnCompressFile( input_file, "bz2" ) )
      {
         cerr << "Unable to uncompress the file " << input_file << "." << endl;
         return false;
      }
      else uncompressed_file_name. SetString( input_file, 0, 4 );

       
   fstream file;
   file. open( uncompressed_file_name. Data(), ios :: in | ios :: binary );
   if( ! file )
   {
      cout << " unable to open file " << uncompressed_file_name. Data() << endl;
      return false;
   }
   if( ! u. Load( file ) )
   {
      cout << " unable to restore the data " << endl;
      return false;
   }
   file. close();
   if( strcmp( input_file + strln - 3, ".gz" ) == 0 &&
       ! CompressFile( uncompressed_file_name. Data(), "gz" ) )
   {
      cerr << "Unable to compress back the file " << input_file << "." << endl;
      return false;
   }
   if( strcmp( input_file + strln - 4, ".bz2" ) == 0 &&
       ! CompressFile( uncompressed_file_name. Data(), "bz2" ) )
   {
      cerr << "Unable to compress back the file " << input_file << "." << endl;
      return false;
   }
   return true;
}

#endif
