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
#include <fstream>

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
      std::list< std::string > elements;

      std::stringstream stream;
};

class Logging
{
public:
   using MetadataElement = std::pair< std::string, std::string >;
   using MetadataMap = std::map< std::string, std::string >;
   using MetadataColumns = std::vector< MetadataElement >;

   using HeaderElements = std::vector< std::string >;
   using RowElements = LoggingRowElements;
   using WidthHints = std::vector< int >;

   Logging( std::ostream& log, int verbose = true )
   : log(log), verbose(verbose)
   {
      try {
         // check if we got an open file
         std::ofstream& file = dynamic_cast< std::ofstream& >( log );
         if( file.is_open() )
            // enable exceptions, but only if we got an open file
            // (under MPI, only the master rank typically opens the log file and thus
            // logs from other ranks are ignored here)
            file.exceptions( std::ostream::failbit | std::ostream::badbit | std::ostream::eofbit );
      }
      catch( std::bad_cast& ) {
         // also enable exceptions if we did not get a file
         log.exceptions( std::ostream::failbit | std::ostream::badbit | std::ostream::eofbit );
      }
   }

   void
   setVerbose( int verbose )
   {
      this->verbose = verbose;
   }

   int getVerbose() const
   {
      return verbose;
   }

   virtual void writeTitle( const std::string& title ) = 0;

   virtual void writeMetadata( const MetadataMap & metadata ) = 0;

   virtual void setMetadataColumns( const MetadataColumns& elements ) = 0;

   virtual void setMetadataElement( const typename MetadataColumns::value_type & element,
                                    int insertPosition = -1 /* negative values insert from the end */ ) = 0;

   virtual void setMetadataWidths( const std::map< std::string, int > & widths ) = 0;

   virtual void
   logResult( const std::string& performer,
              const HeaderElements& headerElements,
              const RowElements& rowElements,
              const WidthHints& columnWidthHints,
              const std::string& errorMessage = "" ) = 0;

   virtual void writeErrorMessage( const std::string& message ) = 0;

   virtual void closeTable() = 0;

protected:
   std::ostream& log;
   int verbose = 0;
};

} // namespace Benchmarks
} // namespace TNL
