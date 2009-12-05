/***************************************************************************
                          tnlCurve.h  -  description
                             -------------------
    begin                : 2007/06/27
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlCurveH
#define tnlCurveH

#include <iomanip>
#include <fstream>
#include "tnlList.h"
#include "tnlObject.h"
#include "mfuncs.h"
#include "compress-file.h"
#include "tnlVector.h"
#include "param-types.h"

//! Basic structure for curves
template< class T > class tnlCurveElement
{
   public:
   tnlCurveElement() {};

   tnlCurveElement( const T& pos, 
                  bool _speparator = false )
      : position( pos ),
        separator( _speparator ) {};
   
   bool Save( ostream& file ) const
   {
      if( ! :: Save( file, position ) ) return false;
      file. write( ( char* ) &separator, sizeof( bool ) );
      if( file. bad() ) return false;
      return true;
   };
   
   bool Load( istream& file ) 
   {
      if( ! :: Load( file, position ) ) return false;
      file. read( ( char* ) &separator, sizeof( bool ) );
      if( file. bad() ) return false;
      return true;
   };
   
   T position;
   
   bool separator;
};

template< class T > class tnlCurve : public tnlObject, public tnlList< tnlCurveElement< T > >
{
   public:
   //! Basic contructor
   tnlCurve()
   { };

   //! Destructor
   ~tnlCurve()
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlCurve< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   //! Append new point
   void Append( const T& vec, bool separator = false )
   {
      tnlList< tnlCurveElement< T > > :: Append( tnlCurveElement< T >( vec, separator ) );
   };

   //! Erase the curve
   void Erase()
   {
      tnlList< tnlCurveElement< T > > :: EraseAll();
   };
   
   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! tnlObject :: Save( file ) ) return false;
      if( ! tnlList< tnlCurveElement< T > > :: DeepSave( file ) ) return false;
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! tnlObject :: Load( file ) ) return false;
      if( ! tnlList< tnlCurveElement< T > > :: DeepLoad( file ) ) return false;
      if( file. bad() ) return false;
      return true;
   };   
};

template< class T > bool Write( const tnlCurve< T >& curve,
                                ostream& str,
                                const char* format,
                                const int step = 1 )
{
   if( ! format )
   {
      cerr << "No format given for drawing 2D grid. " << endl;
      return false;
   }
   if( curve. IsEmpty() )
   {
      cerr << "Unable to draw curve, it's empty!" << endl;
      return false;
   }
   if( strcmp( format, "gnuplot" ) == 0 )
   {
      const int size = curve. Size();
      int i, j;
      for( i = 0; i < size; i += step )
      {
         if( curve[ i ]. separator )
            str << endl;
         else
            str << setprecision( 12 ) 
                << curve[ i ]. position[ 0 ] << " "
                << curve[ i ]. position[ 1 ] << endl;
         for( j = 0; j < step; j ++ )
            if( curve[ i + j ]. separator ) str << endl;
      }
      return true;
   }
   if( strncmp( format, "bin", 3 ) == 0 )
   {
      if( ! curve. Save( str ) ) return false;
      return true;
   }
   cerr << "Unknown format '" << format << "' for drawing a curve." << endl;
   return false;
};

template< class T > bool Write( const tnlCurve< T >& curve,
                                const char* file_name,
                                const char* format,
                                const int step = 1 )
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
   bool result = Write( curve, file, format, step );
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

template< class T > bool Read( tnlCurve< T >& crv,
                               const char* input_file )
{
   int strln = strlen( input_file );
   tnlString uncompressed_file_name( input_file );
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
   if( ! crv. Load( file ) )
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

// Explicit instatiation
template class tnlCurve< tnlVector< 2, double > >;



#endif
