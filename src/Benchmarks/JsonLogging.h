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
   JsonLogging( int verbose = true,
                std::string outputMode = "",
                bool logFileAppend = false )
   : Logging(verbose), outputMode( outputMode ), logFileAppend( logFileAppend )
   {}

   virtual void
   writeTitle( const std::string & title ) override
   {
      if( outputMode == "append" )
         return;

      if( verbose )
         std::cout << std::endl << "== " << title << " ==" << std::endl << std::endl;
   }

   virtual void
   writeMetadata( const MetadataMap & metadata ) override
   {
      if( outputMode == "append" )
         return;

      if( verbose )
         std::cout << "properties:" << std::endl;

      for( auto & it : metadata ) {
         if( verbose )
            std::cout << "   " << it.first << " = " << it.second << std::endl;
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
      this->metadataColumns = elements;
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

   void writeHeader( const HeaderElements& headerElements, const WidthHints& widths )
   {
      TNL_ASSERT_EQ( headerElements.size(), widths.size(), "elements must have equal sizes" );
      if( verbose && header_changed )
      {
         for( auto & lg : metadataColumns )
            std::cout << std::setw( 20 ) << lg.first;
         for( std::size_t i = 0; i < headerElements.size(); i++ )
            std::cout << std::setw( widths[ i ] ) << headerElements[ i ];
         std::cout << std::endl;
         header_changed = false;
      }
   }

   void writeRow( const HeaderElements& headerElements, const RowElements& rowElements, const WidthHints& widths )
   {
      TNL_ASSERT_EQ( headerElements.size(), rowElements.size(), "elements must have equal sizes" );
      TNL_ASSERT_EQ( headerElements.size(), widths.size(), "elements must have equal sizes" );
      if( this->lineStarted )
         log << "," << std::endl;

      log << "      {" << std::endl;

      // write common logs
      int idx( 0 );
      for( auto lg : this->metadataColumns )
      {
         if( verbose )
            std::cout << std::setw( 20 ) << lg.second;
         if( idx++ > 0 )
            log << "," << std::endl;
         log << "         \"" << lg.first << "\" : \"" << lg.second << "\"";
      }

      std::size_t i = 0;
      for( auto el : rowElements )
      {
         if( verbose )
            std::cout << std::setw( widths[ i ] ) << el;
         if( idx++ > 0 )
            log << "," << std::endl;
         log << "         \"" << headerElements[ i ] << "\" : \"" << el << "\"";
         i++;
      }
      log << std::endl << "      }";
      this->lineStarted = true;
      if( verbose )
         std::cout << std::endl;
   }

   virtual void
   logResult( const std::string& spanningElement,
              const HeaderElements& headerElements,
              const RowElements& rowElements,
              const WidthHints& columnWidthHints ) override
   {
      writeHeader( headerElements, columnWidthHints );
      writeRow( headerElements, rowElements, columnWidthHints );
   }

   virtual void
   writeErrorMessage( const std::string& message ) override
   {
      log << "\"error\" : \"" << message << "\"" << std::endl;
   }

   virtual void
   closeTable() override
   {
      header_changed = true;
   }

   virtual bool save( std::ostream & logFile ) override
   {
      if( ! this->logFileAppend )
      {
         logFile << "{" << std::endl;
         logFile << "   \"results\" : [ " << std::endl;
      }
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

   bool lineStarted = false;
   bool logFileAppend = false;
};

} // namespace Benchmarks
} // namespace TNL
