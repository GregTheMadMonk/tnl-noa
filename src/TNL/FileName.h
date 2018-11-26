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

/// \brief Class for creating the full name of a file.
///
/// Merges base name, index number and extention to create the full name of a file.
class FileName
{
   public:

      /// \brief Basic constructor.
      ///
      /// Constructs an empty filename object.
      FileName();
      
      FileName( const String& fileNameBase );
      
      FileName( const String& fileNameBase, 
                const String& extension );
      

      /// \brief Sets the base name of given file.
      ///
      /// Sets \e fileNameBase as the base name of given file.
      /// @param fileNameBase String that specifies new name of file.
      void setFileNameBase( const String& fileNameBase );

      /// \brief Sets the extension of given file.
      ///
      /// Sets \e extension as suffix of a file name.
      /// @param extension A String that specifies extension of file (without dot).
      /// Suffix of a file name. E.g. doc, xls, tnl.
      void setExtension( const String& extension );

      /// \brief Sets index for given file.
      ///
      /// Sets \e index after the base name of given file.
      /// @param index Integer - number of maximum 5(default) digits.
      /// (Number of digits can be changed with \c setDigitsCount).
      void setIndex( const int index );

      /// \brief Sets number of digits for index of given file.
      ///
      /// @param digitsCount Integer - number of digits.
      void setDigitsCount( const int digitsCount );
      
      void setDistributedSystemNodeId( int nodeId );
      
      template< typename Coordinates >
      void setDistributedSystemNodeId( const Coordinates& nodeId );
      
      /// \brief Creates appropriate name for given file.
      ///
      /// Creates particular file name using \e fileNameBase, \e digitsCount,
      /// \e index and \e extension.
      String getFileName();
      
   protected:
   
      String fileNameBase, extension, distributedSystemNodeId;
      
      int index, digitsCount;
   
};

} // namespace TNL

#include <TNL/FileName.hpp>
