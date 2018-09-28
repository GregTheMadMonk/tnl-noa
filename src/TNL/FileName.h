/***************************************************************************
                          FileName.h  -  description
                             -------------------
    begin                : 2007/06/18
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/String.h>

namespace TNL {

String getFileExtension( const String fileName );

void removeFileExtension( String& file_name );

class FileName
{
   public:
      
      FileName();
      
      FileName( const String& fileNameBase );
      
      FileName( const String& fileNameBase, 
                const String& extension );
      
      void setFileNameBase( const String& fileNameBase );
      
      void setExtension( const String& extension );
      
      void setIndex( const int index );
      
      void setDigitsCount( const int digitsCount );
      
      void setDistributedSystemNodeId( int nodeId );
      
      template< typename Coordinates >
      void setDistributedSystemNodeId( const Coordinates& nodeId );
      
      String getFileName();
      
   protected:
   
      String fileNameBase, extension, distributedSystemNodeId;
      
      int index, digitsCount;
   
};

} // namespace TNL

#include <TNL/FileName.hpp>
