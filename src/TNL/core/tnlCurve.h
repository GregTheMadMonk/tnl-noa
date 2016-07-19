/***************************************************************************
                          tnlCurve.h  -  description
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
#include <TNL/core/tnlList.h>
#include <TNL/tnlObject.h>
#include <TNL/core/mfuncs.h>
#include <TNL/core/vectors/tnlStaticVector.h>
#include <TNL/core/param-types.h>

namespace TNL {

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
#ifdef HAVE_NOT_CXX11
      if( ! file. write< const T, tnlHost >( &position ) )
         return false;
      if( ! file. write< const bool, tnlHost >( &separator ) )
         return false;
      return true;
#else
      if( ! file. write( &position ) )
         return false;
      if( ! file. write( &separator ) )
         return false;
      return true;
#endif
   };
 
   bool load( tnlFile& file )
   {
#ifdef HAVE_NOT_CXX11
      if( ! file. read< T, tnlHost >( &position ) )
         return false;
      if( ! file. read< bool, tnlHost >( &separator ) )
         return false;
      return true;
#else
      if( ! file. read( &position ) )
         return false;
      if( ! file. read( &separator ) )
         return false;
      return true;
#endif
   };
 
   T position;
 
   bool separator;
};

template< class T > class tnlCurve : public tnlObject, public tnlList< tnlCurveElement< T > >
{
   public:
   //! Basic contructor
   tnlCurve( const char* name )
   : tnlObject()
// FIXME: name property has been removed from tnlObject
//   : tnlObject( name )
   {
   };

   //! Destructor
   ~tnlCurve()
   { };

   tnlString getType() const
   {
      return tnlString( "tnlCurve< " ) + tnlString( TNL::getType< T >() ) + tnlString( " >" );
   };

   //! Append new point
   void Append( const T& vec, bool separator = false )
   {
      tnlList< tnlCurveElement< T > > :: Append( tnlCurveElement< T >( vec, separator ) );
   };

   //! Erase the curve
   void Erase()
   {
      tnlList< tnlCurveElement< T > >::reset();
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

template< class T > bool Read( tnlCurve< T >& crv,
                               const char* input_file )
{
   tnlFile file;
   if( ! file. open( tnlString( input_file ), tnlReadMode  ) )
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
template class tnlCurve< tnlStaticVector< 2, double > >;

} // namespace TNL

