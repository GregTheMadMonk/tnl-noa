// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <iomanip>

#include <TNL/FileName.h>
#include <TNL/String.h>
#include <TNL/Math.h>

#include "FileName.h"

namespace TNL {

inline FileName::FileName( const String& fileNameBase ) : fileNameBase( fileNameBase )
{}

inline FileName::FileName( const String& fileNameBase, const String& extension )
: fileNameBase( fileNameBase ), extension( extension )
{}

inline void
FileName::setFileNameBase( const String& fileNameBase )
{
   this->fileNameBase = fileNameBase;
}

inline void
FileName::setExtension( const String& extension )
{
   this->extension = extension;
}

inline void
FileName::setIndex( size_t index )
{
   this->index = index;
}

inline void
FileName::setDigitsCount( size_t digitsCount )
{
   this->digitsCount = digitsCount;
}

inline void
FileName::setDistributedSystemNodeId( size_t nodeId )
{
   this->distributedSystemNodeId = "-@";
   this->distributedSystemNodeId += convertToString( nodeId );
}

template< typename Coordinates >
void
FileName::setDistributedSystemNodeCoordinates( const Coordinates& nodeId )
{
   this->distributedSystemNodeId = "-@";
   this->distributedSystemNodeId += convertToString( nodeId[ 0 ] );
   for( int i = 1; i < nodeId.getSize(); i++ ) {
      this->distributedSystemNodeId += "-";
      this->distributedSystemNodeId += convertToString( nodeId[ i ] );
   }
}

inline void
FileName::resetDistributedSystemNodeId()
{
   this->distributedSystemNodeId = "";
}

inline String
FileName::getFileName()
{
   std::stringstream stream;
   stream << this->fileNameBase << std::setw( this->digitsCount ) << std::setfill( '0' ) << this->index
          << this->distributedSystemNodeId << "." << this->extension;
   return stream.str();
}

inline String
getFileExtension( const String& fileName )
{
   const int size = fileName.getLength();
   int i = 1;
   while( fileName[ size - i ] != '.' && i < size )
      i++;
   return fileName.substr( size - i + 1 );
}

inline String
removeFileNameExtension( String fileName )
{
   const int size = fileName.getLength();
   int i = 1;
   while( fileName[ size - i ] != '.' && size > i )
      i++;
   fileName = fileName.substr( 0, size - i );
   return fileName;
}

}  // namespace TNL
