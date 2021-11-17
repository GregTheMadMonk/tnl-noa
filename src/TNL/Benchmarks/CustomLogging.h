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
   // inherit constructors
   using Logging::Logging;

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

   void
   writeTableHeader( const std::string & spanningElement,
                     const HeaderElements & subElements )
   {
      if( verbose && header_changed ) {
         for( auto & it : metadataColumns ) {
            const int width = (metadataWidths.count( it.first )) ? metadataWidths[ it.first ] : 15;
            std::cout << std::setw( width ) << it.first;
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
            const int width = (metadataWidths.count( it.first )) ? metadataWidths[ it.first ] : 15;
            std::cout << std::setw( width ) << it.second;
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
   logResult( const std::string& performer,
              const HeaderElements& headerElements,
              const RowElements& rowElements,
              const WidthHints& columnWidthHints,
              const std::string& errorMessage = "" ) override
   {
      TNL_ASSERT_EQ( headerElements.size(), rowElements.size(), "elements must have equal sizes" );
      TNL_ASSERT_EQ( headerElements.size(), columnWidthHints.size(), "elements must have equal sizes" );
      writeTableHeader( performer, headerElements );
      writeTableRow( performer, rowElements, errorMessage );
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
};

} // namespace Benchmarks
} // namespace TNL
