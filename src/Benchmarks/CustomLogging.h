/***************************************************************************
                          CustomLogging.h  -  description
                             -------------------
    begin                : May 11, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include "Logging.h"
#include <TNL/Assert.h>

namespace TNL {
namespace Benchmarks {

class CustomLogging
: public Logging
{
public:
   CustomLogging( int verbose = true,
                  std::string outputMode = "",
                  bool logFileAppend = false )
   : Logging(verbose), outputMode( outputMode )
   {}

   virtual void
   writeTitle( const std::string & title ) override
   {
      if( verbose )
         std::cout << std::endl << "== " << title << " ==" << std::endl << std::endl;
      log << ": title = " << title << std::endl;
   }

   virtual void
   writeMetadata( const MetadataMap & metadata ) override
   {
      if( verbose )
         std::cout << "properties:" << std::endl;

      for( auto & it : metadata ) {
         if( verbose )
            std::cout << "   " << it.first << " = " << it.second << std::endl;
         log << ": " << it.first << " = " << it.second << std::endl;
      }
      if( verbose )
         std::cout << std::endl;
   }

   virtual void setMetadataColumns( const MetadataColumns& elements ) override
   {
      // check if a header element changed (i.e. a first item of the pairs)
      if( metadataColumns.size() != elements.size() )
         header_changed = true;
      else
         for( std::size_t i = 0; i < metadataColumns.size(); i++ )
            if( metadataColumns[ i ].first != elements[ i ].first ) {
               header_changed = true;
               break;
            }
      metadataColumns = elements;
   }

   virtual void
   setMetadataElement( const typename MetadataColumns::value_type & element,
                       int insertPosition = -1 /* negative values insert from the end */ ) override
   {
      bool found = false;
      for( auto & it : metadataColumns )
         if( it.first == element.first ) {
            if( it.second != element.second )
               it.second = element.second;
            found = true;
            break;
         }
      if( ! found ) {
         if( insertPosition < 0 )
            metadataColumns.insert( metadataColumns.end() + insertPosition + 1, element );
         else
            metadataColumns.insert( metadataColumns.begin() + insertPosition, element );
         header_changed = true;
      }
   }

   void
   writeTableHeader( const std::string & spanningElement,
                     const HeaderElements & subElements )
   {
      if( verbose && header_changed ) {
         for( auto & it : metadataColumns ) {
            std::cout << std::setw( 20 ) << it.first;
         }

         // spanning element is printed as usual column to stdout,
         // but is excluded from header
         std::cout << std::setw( 15 ) << "";

         for( auto & it : subElements ) {
            std::cout << std::setw( 15 ) << it;
         }
         std::cout << std::endl;

         header_changed = false;
      }

      // initial indent string
      log << std::endl;
      for( auto & it : metadataColumns ) {
         log << "! " << it.first << std::endl;
      }

      log << "! " << spanningElement << std::endl;
      for( auto & it : subElements ) {
         log << "!! " << it << std::endl;
      }
   }

   void
   writeTableRow( const std::string & spanningElement,
                  const RowElements & subElements,
                  const std::string & errorMessage )
   {
      if( verbose ) {
         for( auto & it : metadataColumns ) {
            std::cout << std::setw( 20 ) << it.second;
         }
         // spanning element is printed as usual column to stdout
         std::cout << std::setw( 15 ) << spanningElement;
         for( auto & it : subElements ) {
            std::cout << std::setw( 15 ) << it;
         }
         std::cout << std::endl;
      }

      // only when changed (the header has been already adjusted)
      // print each element on separate line
      for( auto & it : metadataColumns ) {
         log << it.second << std::endl;
      }

      if( errorMessage.empty() ) {
         // benchmark data are indented
         const std::string indent = "    ";
         for( auto & it : subElements ) {
            log << indent << it << std::endl;
         }
      }
      else {
         // write the message
         log << errorMessage << std::endl;
      }
   }

   virtual void
   logResult( const std::string& spanningElement,
              const HeaderElements& headerElements,
              const RowElements& rowElements,
              const WidthHints& columnWidthHints,
              const std::string& errorMessage = "" ) override
   {
      TNL_ASSERT_EQ( headerElements.size(), rowElements.size(), "elements must have equal sizes" );
      TNL_ASSERT_EQ( headerElements.size(), columnWidthHints.size(), "elements must have equal sizes" );
      writeTableHeader( spanningElement, headerElements );
      writeTableRow( spanningElement, rowElements, errorMessage );
   }

   virtual void
   writeErrorMessage( const std::string& message ) override
   {
      // initial indent string
      log << std::endl;
      for( auto & it : metadataColumns ) {
         log << "! " << it.first << std::endl;
      }

      // only when changed (the header has been already adjusted)
      // print each element on separate line
      for( auto & it : metadataColumns ) {
         log << it.second << std::endl;
      }

      // write the message
      log << message << std::endl;
   }

   virtual void
   closeTable() override
   {
      log << std::endl;
      header_changed = true;
   }

   virtual bool save( std::ostream & logFile ) override
   {
      closeTable();
      logFile << log.str();
      if( logFile.good() ) {
         log.str() = "";
         return true;
      }
      return false;
   }

protected:
   // manual double -> string conversion with fixed precision
   static std::string
   _to_string( double num, int precision = 0, bool fixed = false )
   {
      std::stringstream str;
      if( fixed )
         str << std::fixed;
      if( precision )
         str << std::setprecision( precision );
      str << num;
      return std::string( str.str().data() );
   }

   std::stringstream log;

   MetadataColumns metadataColumns;
   bool header_changed = true;

   std::string outputMode;
};

} // namespace Benchmarks
} // namespace TNL
