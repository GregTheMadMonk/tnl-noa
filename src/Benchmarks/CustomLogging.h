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

namespace TNL {
namespace Benchmarks {

class CustomLogging
: public Logging
{
public:
   CustomLogging( int verbose = true,
                  String outputMode = "",
                  bool logFileAppend = false )
   : Logging(verbose), outputMode( outputMode )
   {}

   virtual void
   writeTitle( const String & title ) override
   {
      if( verbose )
         std::cout << std::endl << "== " << title << " ==" << std::endl << std::endl;
      log << ": title = " << title << std::endl;
   }

   virtual void addCommonLogs( const CommonLogs& logs ) override
   {
      for( auto log : logs )
      {
         if( verbose )
            std::cout << log.first << " = " << log.second << std::endl;
      }
   };

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

   virtual void
   writeTableHeader( const String & spanningElement,
                     const HeaderElements & subElements ) override
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

      log << header_indent << " " << spanningElement << std::endl;
      for( auto & it : subElements ) {
         log << header_indent << "! " << it << std::endl;
      }
   }

   virtual void
   writeTableRow( const String & spanningElement,
                  const RowElements & subElements ) override
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

   virtual void
   writeErrorMessage( const char* msg ) override
   {
      // initial indent string
      header_indent = "!";
      log << std::endl;
      for( auto & it : metadataColumns ) {
         log << header_indent << " " << it.first << std::endl;
      }

      // only when changed (the header has been already adjusted)
      // print each element on separate line
      for( auto & it : metadataColumns ) {
         log << it.second << std::endl;
      }
      log << msg << std::endl;
   }

   virtual void
   closeTable() override
   {
      log << std::endl;
      header_indent = body_indent = "";
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

   MetadataColumns metadataColumns;
   bool header_changed = true;

   String outputMode;
};

} // namespace Benchmarks
} // namespace TNL
