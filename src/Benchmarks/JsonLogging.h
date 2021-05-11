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

#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include <TNL/String.h>

namespace TNL {
namespace Benchmarks {

class JsonLoggingRowElements
{
   public:

      JsonLoggingRowElements()
      {
         stream << std::setprecision( 6 ) << std::fixed;
      }

      template< typename T >
      JsonLoggingRowElements& operator << ( const T& b )
      {
         stream << b;
         elements.push_back( stream.str() );
         stream.str( std::string() );
         return *this;
      }

      JsonLoggingRowElements& operator << ( decltype( std::setprecision( 2 ) )& setprec )
      {
         stream << setprec;
         return *this;
      }

      JsonLoggingRowElements& operator << ( decltype( std::fixed )& setfixed ) // the same works also for std::scientific
      {
         stream << setfixed;
         return *this;
      }

      // iterators
      auto begin() noexcept { return elements.begin(); }

      auto begin() const noexcept { return elements.begin(); }

      auto cbegin() const noexcept { return elements.cbegin(); }

      auto end() noexcept { return elements.end(); }

      auto end() const noexcept { return elements.end(); }

      auto cend() const noexcept { return elements.cend(); }

   protected:
      std::list< String > elements;

      std::stringstream stream;
};

class JsonLogging
{
public:
   using MetadataElement = std::pair< const char*, String >;
   using MetadataMap = std::map< const char*, String >;
   using MetadataColumns = std::vector<MetadataElement>;

   using HeaderElements = std::vector< String >;
   using RowElements = LoggingRowElements;

   JsonLogging( int verbose = true )
   : verbose(verbose)
   {}

   void
   setVerbose( int verbose)
   {
      this->verbose = verbose;
   }

   void
   writeTitle( const String & title )
   {
      if( verbose )
         std::cout << std::endl << "== " << title << " ==" << std::endl << std::endl;
      log << ": title = " << title << std::endl;
   }

   void
   writeMetadata( const MetadataMap & metadata )
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
   writeTableHeader( const String & spanningElement,
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
      header_indent = "!";
      log << std::endl;
      for( auto & it : metadataColumns ) {
         log << header_indent << " " << it.first << std::endl;
      }

      // dump stacked spanning columns
      if( horizontalGroups.size() > 0 )
         while( horizontalGroups.back().second <= 0 ) {
            horizontalGroups.pop_back();
            header_indent.pop_back();
         }
      for( size_t i = 0; i < horizontalGroups.size(); i++ ) {
         if( horizontalGroups[ i ].second > 0 ) {
            log << header_indent << " " << horizontalGroups[ i ].first << std::endl;
            header_indent += "!";
         }
      }

      log << header_indent << " " << spanningElement << std::endl;
      for( auto & it : subElements ) {
         log << header_indent << "! " << it << std::endl;
      }

      if( horizontalGroups.size() > 0 ) {
         horizontalGroups.back().second--;
         header_indent.pop_back();
      }
   }

   void
   writeTableRow( const String & spanningElement,
                  const RowElements & subElements )
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

      // benchmark data are indented
      const String indent = "    ";
      for( auto & it : subElements ) {
         log << indent << it << std::endl;
      }
   }

   void
   writeErrorMessage( const char* msg,
                      int colspan = 1 )
   {
      // initial indent string
      header_indent = "!";
      log << std::endl;
      for( auto & it : metadataColumns ) {
         log << header_indent << " " << it.first << std::endl;
      }

      // make sure there is a header column for the message
      if( horizontalGroups.size() == 0 )
         horizontalGroups.push_back( {"", 1} );

      // dump stacked spanning columns
      while( horizontalGroups.back().second <= 0 ) {
         horizontalGroups.pop_back();
         header_indent.pop_back();
      }
      for( size_t i = 0; i < horizontalGroups.size(); i++ ) {
         if( horizontalGroups[ i ].second > 0 ) {
            log << header_indent << " " << horizontalGroups[ i ].first << std::endl;
            header_indent += "!";
         }
      }
      if( horizontalGroups.size() > 0 ) {
         horizontalGroups.back().second -= colspan;
         header_indent.pop_back();
      }

      // only when changed (the header has been already adjusted)
      // print each element on separate line
      for( auto & it : metadataColumns ) {
         log << it.second << std::endl;
      }
      log << msg << std::endl;
   }

   void
   closeTable()
   {
      log << std::endl;
      header_indent = body_indent = "";
      header_changed = true;
      horizontalGroups.clear();
   }

   bool save( std::ostream & logFile )
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
   // manual double -> String conversion with fixed precision
   static String
   _to_string( double num, int precision = 0, bool fixed = false )
   {
      std::stringstream str;
      if( fixed )
         str << std::fixed;
      if( precision )
         str << std::setprecision( precision );
      str << num;
      return String( str.str().data() );
   }

   std::stringstream log;
   std::string header_indent;
   std::string body_indent;

   int verbose;
   MetadataColumns metadataColumns;
   bool header_changed = true;
   std::vector< std::pair< String, int > > horizontalGroups;
};

} // namespace Benchmarks
} // namespace TNL
