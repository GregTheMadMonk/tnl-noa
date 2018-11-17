/***************************************************************************
                          FileName.hpp  -  description
                             -------------------
    begin                : Sep 28, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <sstream>
#include <iomanip>

#include <TNL/FileName.h>
#include <TNL/String.h>
#include <TNL/Math.h>

namespace TNL {

inline FileName::FileName()
: index( 0 ), digitsCount( 5 )
{
}

inline FileName::FileName( const String& fileNameBase )
: fileNameBase( fileNameBase ),
  index( 0 ),
  digitsCount( 5 )
{
}

inline FileName::FileName( const String& fileNameBase,
                           const String& extension )
: fileNameBase( fileNameBase ),
  extension( extension ),
  index( 0 ),
  digitsCount( 5 )
{
}

inline void FileName::setFileNameBase( const String& fileNameBase )
{
   this->fileNameBase = fileNameBase;
}

inline void FileName::setExtension( const String& extension )
{
   this->extension = extension;
}

inline void FileName::setIndex( const int index )
{
   this->index = index;
}

inline void FileName::setDigitsCount( const int digitsCount )
{
   this->digitsCount = digitsCount;
}

inline void FileName::setDistributedSystemNodeId( int nodeId )
{
   this->distributedSystemNodeId = "-";
   this->distributedSystemNodeId += convertToString( nodeId );
}

template< typename Coordinates >
void
FileName::
setDistributedSystemNodeId( const Coordinates& nodeId )
{
   this->distributedSystemNodeId = "-";
   this->distributedSystemNodeId += convertToString( nodeId[ 0 ] );
   for( int i = 1; i < nodeId.getSize(); i++ )
   {
      this->distributedSystemNodeId += "-";
      this->distributedSystemNodeId += convertToString( nodeId[ i ] );
   }
}

inline String FileName::getFileName()
{
   std::stringstream stream;
   stream << this->fileNameBase
          << std::setw( this->digitsCount )
          << std::setfill( '0' )
          << this->index
          << this->distributedSystemNodeId
          << "." << this->extension;
   return String( stream.str().data() );
}

inline String getFileExtension( const String fileName )
{
   int size = fileName. getLength();
   int i = 1;
   while( fileName. getString()[ size - i ] != '.' && size > i  ) i ++ ;
   String result;
   result.setString( fileName. getString(), size - i + 1 );
   return result;
}

inline void removeFileExtension( String& fileName )
{
   int size = fileName. getLength();
   int i = 1;
   while( fileName. getString()[ size - i ] != '.' && size > i  ) i ++ ;
   fileName. setString( fileName. getString(), 0, i );
}

} // namespace TNL
