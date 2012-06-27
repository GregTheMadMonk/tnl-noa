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
#include <core/tnlList.h>
#include <core/tnlObject.h>
#include <core/mfuncs.h>
#include <core/tnlTuple.h>
#include <core/param-types.h>

//! Basic structure for curves
template< class T > class tnlCurveElement
{
   public:
   tnlCurveElement() {};

   tnlCurveElement( const T& pos, 
                  bool _speparator = false )
      : position( pos ),
        separator( _speparator ) {};
   
   bool save( tnlFile& file ) const
   {
      if( ! file. write( &position ) )
         return false;
      if( ! file. write( &separator ) )
         return false;
      return true;
   };
   
   bool load( tnlFile& file )
   {
      if( ! file. read( &position ) )
         return false;
      if( ! file. read( &separator ) )
         return false;
      return true;
   };
   
   T position;
   
   bool separator;
};

template< class T > class tnlCurve : public tnlObject, public tnlList< tnlCurveElement< T > >
{
   public:
   //! Basic contructor
   tnlCurve( const char* name )
   : tnlObject( name )
   {
   };

   //! Destructor
   ~tnlCurve()
   { };

   tnlString getType() const
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
   bool save( tnlFile& file ) const
   {
      if( ! tnlObject :: save( file ) ) return false;
      if( ! tnlList< tnlCurveElement< T > > :: DeepSave( file ) ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool load( tnlFile& file )
   {
      if( ! tnlObject :: load( file ) ) return false;
      if( ! tnlList< tnlCurveElement< T > > :: DeepLoad( file ) ) return false;
      return true;
   };

   //! Method for saving the object to a file as a binary data
   bool save( const tnlString& fileName ) const
   {
      return tnlObject :: save( fileName );
   };

   //! Method for restoring the object from a file
   bool load( const tnlString& fileName )
   {
      return tnlObject :: load( fileName );
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
   if( curve. isEmpty() )
   {
      cerr << "Unable to draw curve, it's empty!" << endl;
      return false;
   }
   if( strcmp( format, "gnuplot" ) == 0 )
   {
      const int size = curve. getSize();
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
   cerr << "Unknown format '" << format << "' for drawing a curve." << endl;
   return false;
};

template< class T > bool Write( const tnlCurve< T >& curve,
                                const char* file_name,
                                const char* format,
                                const int step = 1 )
{

   if( strncmp( format, "tnl",3 ) == 0 )
   {
      tnlFile file;
      if( ! file. open( tnlString( file_name ) + tnlString( ".tnl" ), tnlWriteMode ) )
      {
         cerr << "I am not able to open the file " << file_name << " for drawing curve "
              << curve. getName() <<"." << endl;
         return false;
      }
      if( ! curve. save( file ) )
      {
         cerr << "I am not able to write to the file " << file_name << " for drawing grid "
              << curve. getName() <<"." << endl;
         return false;
      }
      file. close();
   }
   else
   {
      fstream file;
      file. open( file_name, ios :: out );
      if( ! file )
      {
         cerr << "I am not able to to open the file " << file_name << " for drawing curve "
              << curve. getName() <<"." << endl;
         return false;
      }
      bool result = Write( curve, file, format, step );
      file. close();
      if( ! result )
      {
         cerr << "Sorry I could not write to the file " << file_name << endl;
         return false;
      }
   }
   return true;
};

template< class T > bool Read( tnlCurve< T >& crv,
                               const char* input_file )
{
   tnlFile file;
   if( ! file. open( tnlString( input_file ), tnlReadMode  ) )
   {
      cout << " unable to open file " << input_file << endl;
      return false;
   }
   if( ! crv. load( file ) )
   {
      cout << " unable to restore the data " << endl;
      return false;
   }
   file. close();
   return true;
}

// Explicit instatiation
template class tnlCurve< tnlTuple< 2, double > >;



#endif
