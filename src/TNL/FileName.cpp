/***************************************************************************
                          FileName.cpp  -  description
                             -------------------
    begin                : 2007/06/18
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <sstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <TNL/FileName.h>
#include <TNL/String.h>
#include <TNL/Math.h>

namespace TNL {
   
FileName::FileName()
: index( 0 ), digitsCount( 5 )
{
}

FileName::FileName( const String& fileNameBase )
: fileNameBase( fileNameBase ),
   index( 0 ),
   digitsCount( 5 )
{
}

FileName::FileName( const String& fileNameBase, 
                    const String& extension )
: fileNameBase( fileNameBase ),
   extension( extension ),
   index( 0 ),
   digitsCount( 5 )
{
}
      
void FileName::setFileNameBase( const String& fileNameBase )
{
   this->fileNameBase = fileNameBase;
}
      
void FileName::setExtension( const String& extension )
{
   this->extension = extension;
}
    
void FileName::setIndex( const int index )
{
   this->index = index;
}
    
void FileName::setDigitsCount( const int digitsCount )
{
   this->digitsCount = digitsCount;
}

void FileName::setDistributedSystemNodeId( int nodeId )
{
   this->distributedSystemNodeId = "-";
   this->distributedSystemNodeId += convertToString( nodeId );
}

String FileName::getFileName()
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

String getFileExtension( const String fileName )
{
   int size = fileName. getLength();
   int i = 1;
   while( fileName. getString()[ size - i ] != '.' && size > i  ) i ++ ;
   String result;
   result.setString( fileName. getString(), size - i + 1 );
   return result;
}

void removeFileExtension( String& fileName )
{
   int size = fileName. getLength();
   int i = 1;
   while( fileName. getString()[ size - i ] != '.' && size > i  ) i ++ ;
   fileName. setString( fileName. getString(), 0, i );
}

} // namespace TNL