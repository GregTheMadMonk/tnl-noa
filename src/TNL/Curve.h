/***************************************************************************
                          Curve.h  -  description
                             -------------------
    begin                : 2007/06/27
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <fstream>
#include <cstring>
#include <TNL/Containers/List.h>
#include <TNL/Object.h>
#include <TNL/Math.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/param-types.h>

namespace TNL {

//! Basic structure for curves
template< class T >
class CurveElement
{
   public:
   CurveElement() {};

   CurveElement( const T& pos,
                  bool _speparator = false )
      : position( pos ),
        separator( _speparator ) {};
 
   bool save( File& file ) const
   {
      if( ! file. write( &position ) )
         return false;
      if( ! file. write( &separator ) )
         return false;
      return true;
   };
 
   bool load( File& file )
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

template< class T >
class Curve
 : public Object,
   public Containers::List< CurveElement< T > >
{
   public:
   //! Basic contructor
   Curve( const char* name )
   : Object()
// FIXME: name property has been removed from Object
//   : Object( name )
   {
   };

   //! Destructor
   ~Curve()
   { };

   String getType() const
   {
      return String( "Curve< " ) + String( TNL::getType< T >() ) + String( " >" );
   };

   //! Append new point
   void Append( const T& vec, bool separator = false )
   {
      Containers::List< CurveElement< T > > :: Append( CurveElement< T >( vec, separator ) );
   };

   //! Erase the curve
   void Erase()
   {
      Containers::List< CurveElement< T > >::reset();
   };
 
   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const
   {
      if( ! Object :: save( file ) ) return false;
      if( ! Containers::List< CurveElement< T > > :: DeepSave( file ) ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool load( File& file )
   {
      if( ! Object :: load( file ) ) return false;
      if( ! Containers::List< CurveElement< T > > :: DeepLoad( file ) ) return false;
      return true;
   };

   //! Method for saving the object to a file as a binary data
   bool save( const String& fileName ) const
   {
      return Object :: save( fileName );
   };

   //! Method for restoring the object from a file
   bool load( const String& fileName )
   {
      return Object :: load( fileName );
   };

};

template< class T > bool Write( const Curve< T >& curve,
                                std::ostream& str,
                                const char* format,
                                const int step = 1 )
{
   if( ! format )
   {
      std::cerr << "No format given for drawing 2D grid. " << std::endl;
      return false;
   }
   if( curve. isEmpty() )
   {
      std::cerr << "Unable to draw curve, it's empty!" << std::endl;
      return false;
   }
   if( strcmp( format, "gnuplot" ) == 0 )
   {
      const int size = curve. getSize();
      int i, j;
      for( i = 0; i < size; i += step )
      {
         if( curve[ i ]. separator )
            str << std::endl;
         else
            str << std::setprecision( 12 )
                << curve[ i ]. position[ 0 ] << " "
                << curve[ i ]. position[ 1 ] << std::endl;
         for( j = 0; j < step; j ++ )
            if( curve[ i + j ]. separator ) str << std::endl;
      }
      return true;
   }
   std::cerr << "Unknown format '" << format << "' for drawing a curve." << std::endl;
   return false;
};

template< class T > bool Write( const Curve< T >& curve,
                                const char* file_name,
                                const char* format,
                                const int step = 1 )
{

   if( strncmp( format, "tnl",3 ) == 0 )
   {
      File file;
      if( ! file. open( String( file_name ) + String( ".tnl" ), tnlWriteMode ) )
      {
         std::cerr << "I am not able to open the file " << file_name << " for drawing curve." << std::endl;
         return false;
      }
      if( ! curve. save( file ) )
      {
         std::cerr << "I am not able to write to the file " << file_name << " for drawing grid." << std::endl;
         return false;
      }
      file. close();
   }
   else
   {
      std::fstream file;
      file. open( file_name, std::ios::out );
      if( ! file )
      {
         std::cerr << "I am not able to to open the file " << file_name << " for drawing curve." << std::endl;
         return false;
      }
      bool result = Write( curve, file, format, step );
      file. close();
      if( ! result )
      {
         std::cerr << "Sorry I could not write to the file " << file_name << std::endl;
         return false;
      }
   }
   return true;
};

template< class T > bool Read( Curve< T >& crv,
                               const char* input_file )
{
   File file;
   if( ! file. open( String( input_file ), tnlReadMode  ) )
   {
     std::cout << " unable to open file " << input_file << std::endl;
      return false;
   }
   if( ! crv. load( file ) )
   {
     std::cout << " unable to restore the data " << std::endl;
      return false;
   }
   file. close();
   return true;
}

// Explicit instatiation
template class Curve< Containers::StaticVector< 2, double > >;

} // namespace TNL

