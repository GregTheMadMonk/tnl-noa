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

#include "FileName.h"

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

inline void FileName::setIndex( const size_t index )
{
   this->index = index;
}

inline void FileName::setDigitsCount( const size_t digitsCount )
{
   this->digitsCount = digitsCount;
}

inline void FileName::setDistributedSystemNodeId( size_t nodeId )
{
   this->distributedSystemNodeId = "-@";
   this->distributedSystemNodeId += convertToString( nodeId );
}

template< typename Coordinates >
void
FileName::
setDistributedSystemNodeCoordinates( const Coordinates& nodeId )
{
   this->distributedSystemNodeId = "-@";
   this->distributedSystemNodeId += convertToString( nodeId[ 0 ] );
   for( int i = 1; i < nodeId.getSize(); i++ )
   {
      this->distributedSystemNodeId += "-";
      this->distributedSystemNodeId += convertToString( nodeId[ i ] );
   }
}

void
FileName::
resetDistributedSystemNodeId()
{
   this->distributedSystemNodeId = "";
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
   const int size = fileName.getLength();
   int i = 1;
   while( fileName[ size - i ] != '.' && i < size ) i++;
   return fileName.substr( size - i + 1 );
}

inline String removeFileNameExtension( String fileName )
{
   const int size = fileName.getLength();
   int i = 1;
   while( fileName[ size - i ] != '.' && size > i ) i++;
   fileName = fileName.substr( 0, size - i + 1 );
   return fileName;
}

} // namespace TNL
