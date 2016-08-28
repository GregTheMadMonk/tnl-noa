/***************************************************************************
                          mfilename.h  -  description
                             -------------------
    begin                : 2007/06/18
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/String.h>

namespace TNL {

/*void FileNameBaseNumberEnding( const char* base_name,
                               int number,
                               int index_size,
                               const char* ending,
                               String& file_name );*/

String getFileExtension( const String fileName );

void removeFileExtension( String& file_name );

class FileName
{
   public:
      
      FileName();
      
      void setFileNameBase( const String& fileNameBase );
      
      void setExtension( const String& extension );
      
      void setIndex( const int index );
      
      void setDigitsCount( const int digitsCount );
      
      String getFileName();
      
   protected:
   
      String fileNameBase, extension;
      
      int index, digitsCount;
   
};

} // namespace TNL
