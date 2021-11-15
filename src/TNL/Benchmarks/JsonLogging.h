/***************************************************************************
                          JsonLogging.h  -  description
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

class JsonLogging
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
   }

   virtual void
   writeMetadata( const MetadataMap & metadata ) override
   {
      if( verbose ) {
         std::cout << "properties:" << std::endl;
         for( auto & it : metadata )
            std::cout << "   " << it.first << " = " << it.second << std::endl;
         std::cout << std::endl;
      }
   }

   void writeHeader( const HeaderElements& headerElements, const WidthHints& widths )
   {
      TNL_ASSERT_EQ( headerElements.size(), widths.size(), "elements must have equal sizes" );
      if( verbose && header_changed )
      {
         for( auto & lg : metadataColumns ) {
            const int width = (metadataWidths.count( lg.first )) ? metadataWidths[ lg.first ] : 14;
            std::cout << std::setw( width ) << lg.first;
         }
         for( std::size_t i = 0; i < headerElements.size(); i++ )
            std::cout << std::setw( widths[ i ] ) << headerElements[ i ];
         std::cout << std::endl;
         header_changed = false;
      }
   }

   void writeRow( const HeaderElements& headerElements,
                  const RowElements& rowElements,
                  const WidthHints& widths,
                  const std::string& errorMessage )
   {
      TNL_ASSERT_EQ( headerElements.size(), rowElements.size(), "elements must have equal sizes" );
      TNL_ASSERT_EQ( headerElements.size(), widths.size(), "elements must have equal sizes" );

      log << "{";

      // write common logs
      int idx( 0 );
      for( auto lg : this->metadataColumns )
      {
         if( verbose ) {
            const int width = (metadataWidths.count( lg.first )) ? metadataWidths[ lg.first ] : 14;
            std::cout << std::setw( width ) << lg.second;
         }
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << lg.first << "\": \"" << lg.second << "\"";
      }

      std::size_t i = 0;
      for( auto el : rowElements )
      {
         if( verbose )
            std::cout << std::setw( widths[ i ] ) << el;
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << headerElements[ i ] << "\": \"" << el << "\"";
         i++;
      }
      if( ! errorMessage.empty() ) {
         if( idx++ > 0 )
            log << ", ";
         log << "\"error\": \"" << errorMessage << "\"";
      }
      log << "}" << std::endl;
      if( verbose )
         std::cout << std::endl;
   }

   virtual void
   logResult( const std::string& performer,
              const HeaderElements& headerElements,
              const RowElements& rowElements,
              const WidthHints& columnWidthHints,
              const std::string& errorMessage = "" ) override
   {
      setMetadataElement({ "performer", performer });
      writeHeader( headerElements, columnWidthHints );
      writeRow( headerElements, rowElements, columnWidthHints, errorMessage );
   }

   virtual void
   writeErrorMessage( const std::string& message ) override
   {
      log << "{";

      // write common logs
      int idx( 0 );
      for( auto lg : this->metadataColumns )
      {
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << lg.first << "\": \"" << lg.second << "\"";
      }

      if( idx++ > 0 )
         log << ", ";
      log << "\"error\": \"" << message << "\"";

      log << "}" << std::endl;
   }

   virtual void
   closeTable() override
   {
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
