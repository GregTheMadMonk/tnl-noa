/***************************************************************************
                          drawGrid3D.h  -  description
                             -------------------
    begin                : 2009/07/26
    copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef drawGrid3DH
#define drawGrid3DH

#include <ostream>
#include <fstream>
#include <iomanip>
#include <float.h>
#include <diff/mGrid3D.h>

template< class T > bool Draw( const mGrid3D< T >& u,
                               ostream& str,
                               const char* format,
                               const long int i_step = 1,
                               const long int j_step = 1,
                               const long int k_step = 1 )
{
   if( ! format )
   {
      cerr << "No format given for drawing 3D grid. " << endl;
      return false;
   }


   long int i, j, k;
   const long int x_size = u. GetXSize();
   const long int y_size = u. GetYSize();
   const long int z_size = u. GetZSize();
   const double& ax = u. GetAx();
   const double& ay = u. GetAy();
   const double& az = u. GetAy();
   const double& hx = u. GetHx();
   const double& hy = u. GetHy();
   const double& hz = u. GetHy();
   if( strncmp( format, "bin", 3 ) == 0 )
   {
      if( ! u. Save( str ) ) return false;
      return true;
   }
   if( strcmp( format, "gnuplot" ) == 0 )
   {
      cout << "GNUPLOT is not supported for mGrid3D." << endl;
      return false;
   }
   if( strcmp( format, "vti" ) == 0 )
   {
      str << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
      str << "<ImageData WholeExtent=\"" 
           << "0 " << x_size - 1 
           << " 0 " << y_size - 1
           << " 0 " << z_size - 1 << "\" Origin=\"0 0 0\" Spacing=\"" 
           << hx << " " << hy << " " << hz << "\">" << endl;
      str << "<Piece Extent=\"0 "
          << x_size - 1 << " 0 " 
          << y_size - 1 << " 0 "
          << z_size - 1 << "\">" << endl;
      str << "<PointData Scalars=\"order_parameter\">" << endl;
      str << "<DataArray Name=\"order_parameter\" type=\"Float32\" format=\"ascii\">" << endl;
      str. flags( ios_base::scientific );
      long int i, j, k;
      for( k = 0; k <= z_size - k_step; k += k_step )
         for( j = 0; j <= y_size - j_step; j += j_step )
            for( i = 0; i <= x_size - i_step; i += i_step )
            {
              str << u( i, j, k ) << " ";
            }
      str << endl;
      str << "</DataArray>" << endl;
      str << "</PointData>" << endl;
      str << "</Piece>" << endl;
      str << "</ImageData>" << endl;
      str << "</VTKFile>" << endl;
      return true;
   }
   if( strcmp( format, "povray" ) == 0 )
   {
      str. put( ( char ) ( x_size >> 8 ) );
      str. put( ( char ) ( x_size & 0xff ) );
      str. put( ( char ) ( y_size >> 8 ) );
      str. put( ( char ) ( y_size & 0xff ) );
      str. put( ( char ) ( z_size >> 8 ) );
      str. put( ( char ) ( z_size & 0xff ) );
      long int i, j, k;
      double min( DBL_MAX ), max( -DBL_MAX );
      for( k = 0; k < z_size; k ++ ) 
         for( j = 0; j < y_size; j ++ ) 
            for( i = 0; i < x_size; i ++ )
            {
               min = Min( min, u( i, j, k ) );
               max = Max( max, u( i, j, k ) );
            }
            
      for( k = 0; k < z_size; k ++ ) 
         for( j = 0; j < y_size; j ++ ) 
            for( i = 0; i < x_size; i ++ )
            {
               int v = 255.0 * ( u( i, j, k ) - min ) / ( max - min );
               str. write( ( char* ) &v, sizeof( int ) );
            }
      return true;
   }
   cerr << "Unknown format '" << format << "' for drawing a grid 3D." << endl;
   return false;
};

template< class T > bool Draw( const mGrid3D< T >& u,
                               const char* file_name,
                               const char* format,
                               const long int i_step = 1,
                               const long int j_step = 1,
                               const long int k_step = 1 )
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
   bool result = Draw( u, file, format, i_step, j_step, k_step );
   file. close();
   if( ! result )
      return false;
   int len = strlen( format );
   if( strcmp( format + Max( 0, len - 3 ), "-gz" ) == 0 && 
       ! CompressFile( file_name, "gz" ) )
      return false;
   if( strcmp( format + Max( 0, len - 4 ), "-bz2" ) == 0  &&
       ! CompressFile( file_name, "bz2" ) )
         return false;
   return true;
};

template< class T > bool Read( mGrid3D< T >& u,
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
