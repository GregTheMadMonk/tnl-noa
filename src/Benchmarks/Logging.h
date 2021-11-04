/***************************************************************************
                          Logging.h  -  description
                             -------------------
    begin                : Dec 25, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include <list>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include <TNL/String.h>

namespace TNL {
namespace Benchmarks {

class LoggingRowElements
{
   public:

      LoggingRowElements()
      {
         stream << std::setprecision( 6 ) << std::fixed;
      }

      template< typename T >
      LoggingRowElements& operator << ( const T& b )
      {
         stream << b;
         elements.push_back( stream.str() );
         stream.str( std::string() );
         return *this;
      }

      LoggingRowElements& operator << ( decltype( std::setprecision( 2 ) )& setprec )
      {
         stream << setprec;
         return *this;
      }

      LoggingRowElements& operator << ( decltype( std::fixed )& setfixed ) // the same works also for std::scientific
      {
         stream << setfixed;
         return *this;
      }

      std::size_t size() const noexcept { return elements.size(); };

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

class Logging
{
public:
   using MetadataElement = std::pair< const char*, String >;
   using MetadataMap = std::map< const char*, String >;
   using MetadataColumns = std::vector<MetadataElement>;

   using HeaderElements = std::vector< String >;
   using RowElements = LoggingRowElements;

   using CommonLogs = std::vector< std::pair< const char*, String > >;
   using LogsMetadata = HeaderElements;
   using WidthHints = std::vector< int >;

   Logging( int verbose = true )
   : verbose(verbose)
   {}

   void
   setVerbose( int verbose )
   {
      this->verbose = verbose;
   }

   int getVerbose() const
   {
      return verbose;
   }

   virtual void writeTitle( const String & title ) = 0;

   virtual void addCommonLogs( const CommonLogs& logs ) = 0;

   virtual void resetLogsMetada() {}

   virtual void addLogsMetadata( const LogsMetadata& md, const WidthHints& widths ) {}

   virtual void writeHeader() {}

   virtual void writeMetadata( const MetadataMap & metadata ) {}

   virtual void
   writeTableHeader( const String & spanningElement,
                     const HeaderElements & subElements ) = 0;

   virtual void
   writeTableRow( const String & spanningElement,
                  const RowElements & subElements ) = 0;

   virtual void
   writeErrorMessage( const char* msg ) = 0;

   virtual void closeTable() = 0;

   virtual bool save( std::ostream & logFile ) = 0;

protected:
   int verbose = 0;
};

} // namespace Benchmarks
} // namespace TNL
