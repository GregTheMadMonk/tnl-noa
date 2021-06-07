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

      size_t size() const noexcept { return this->elements.size(); };
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

   using CommonLogs = std::vector< std::pair< const char*, String > >;
   using LogsMetadata = std::vector< String >;

   using HeaderElements = std::vector< String >;
   using RowElements = JsonLoggingRowElements;

   JsonLogging( int verbose = true,
                String outputMode = "",
                bool logFileAppend = false )
   : verbose(verbose), outputMode( outputMode ), logFileAppend( logFileAppend )
   {}

   void
   setVerbose( int verbose)
   {
      this->verbose = verbose;
   }

   void addCommonLogs( const CommonLogs& logs )
   {
      //if( this->lineStarted )
      //   log << "," << std::endl;
      //log << "   \"benchmarks\" : [" << std::endl;
      this->commonLogs = logs;
      //int idx( 0 );
      for( auto lg : logs )
      {
         if( verbose )
            std::cout << lg.first << " = " << lg.second << std::endl;
         //if( idx++ > 0 )
         //   log << "," << std::endl;
         //log << "      \"" << lg.first << "\" : \"" << lg.second << "\"";
         //this->lineStarted = true;
      }
   };

   void resetLogsMetadat() { this->logsMetadata.clear(); };

   void addLogsMetadata( const std::vector< String >& md )
   {
      this->logsMetadata.insert( this->logsMetadata.end(), md.begin(), md.end() );
   }

   void writeHeader()
   {
      if( verbose )
      {
         for( auto md : this->logsMetadata )
            std::cout << md << "\t";
         std::cout << std::endl;
      }
   }

   void writeRow( const RowElements& rowEls )
   {
      TNL_ASSERT_EQ( rowEls.size(), this->logsMetadata.size(), "" );
      if( this->lineStarted )
         log << "," << std::endl;

      log << "         {" << std::endl;

      // write common logs
      int idx( 0 );
      for( auto lg : this->commonLogs )
      {
         //if( verbose )
         //   std::cout << lg.first << " = " << lg.second << std::endl;
         if( idx++ > 0 )
            log << "," << std::endl;
         log << "      \"" << lg.first << "\" : \"" << lg.second << "\"";
      }

      auto md = this->logsMetadata.begin();
      for( auto el : rowEls )
      {
         if( verbose )
            std::cout << el << "\t";
         if( idx++ > 0 )
            log << "," << std::endl;
         log << "          \"" << *md++ << "\" : \"" << el << "\"";
      }
      log << std::endl << "         }";
      this->lineStarted = true;
      if( verbose )
         std::cout << std::endl;
   }

   void
   writeTitle( const String & title )
   {
      if( outputMode == "append" )
         return;

      if( verbose )
         std::cout << std::endl << "== " << title << " ==" << std::endl << std::endl;
      log << "   \"title\" : \"" << title << "\"";
      this->lineStarted = true;
   }

   void
   writeMetadata( const MetadataMap & metadata )
   {
      if( outputMode == "append" )
      {
         this->lineStarted = true;
         return;
      }

      if( verbose )
         std::cout << "properties:" << std::endl;

      int idx( this->lineStarted );
      for( auto & it : metadata ) {
         if( verbose )
            std::cout << "   " << it.first << " = " << it.second << std::endl;
         if( idx++ > 0 )
            log << "," << std::endl;
         log << "   \"" << it.first << "\" : \"" << it.second << "\"";
         //this->lineStarted = true;
      }
      log << "," << std::endl << "      \"results\" : [ " << std::endl;
      this->lineStarted = false;

      if( verbose )
         std::cout << std::endl;
   }

   void
   writeTableHeader( const String & spanningElement,
                     const HeaderElements & subElements )
   {
   }

   void
   writeTableRow( const String & spanningElement,
                  const RowElements & subElements )
   {
      writeRow( subElements );
   }

   void
   writeErrorMessage( const char* msg,
                      int colspan = 1 )
   {
      log << "\"error\" : \"" << msg << "\"" << std::endl;
      // initial indent string
      /*header_indent = "!";
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
      */
   }

   void
   closeTable()
   {
      //log << std::endl << "   ]" << std::endl;
      //log << "," << std::endl;
      //header_indent = body_indent = "";
      //header_changed = true;
      //horizontalGroups.clear();
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

   // new JSON implementation
   LogsMetadata logsMetadata;
   CommonLogs commonLogs;
   String outputMode;

   bool lineStarted = false;
   bool resultsStarted = false;
   bool logFileAppend = false;
};

} // namespace Benchmarks
} // namespace TNL
