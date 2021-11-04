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

   virtual void addCommonLogs( const CommonLogs& logs ) override
   {
      this->commonLogs = logs;
      if( verbose )
      {
         std::cout << std::endl << "Benchmark setup:" << std::endl;
         for( auto lg : logs )
            std::cout << "   " << lg.first << " = " << lg.second << std::endl;
         std::cout << std::endl;
      }
   };

   virtual void resetLogsMetada() override
   {
      this->logsMetadata.clear();
      this->widthHints.clear();
   }

   virtual void addLogsMetadata( const LogsMetadata& md, const WidthHints& widths ) override
   {
      this->logsMetadata.insert( this->logsMetadata.end(), md.begin(), md.end() );
      this->widthHints.insert( this->widthHints.end(), widths.begin(), widths.end() );
   }

   virtual void writeHeader() override
   {
      TNL_ASSERT_EQ( this->logsMetadata.size(), this->widthHints.size(), "" );
      if( verbose )
      {
         for( std::size_t i = 0; i < this->logsMetadata.size(); i++ )
            std::cout << std::setw( this->widthHints[ i ] ) << this->logsMetadata[ i ];
         std::cout << std::endl;
      }
   }

   void writeRow( const RowElements& rowEls )
   {
      TNL_ASSERT_EQ( rowEls.size(), this->logsMetadata.size(), "" );
      if( this->lineStarted )
         log << "," << std::endl;

      log << "      {" << std::endl;

      // write common logs
      int idx( 0 );
      for( auto lg : this->commonLogs )
      {
         if( idx++ > 0 )
            log << "," << std::endl;
         log << "         \"" << lg.first << "\" : \"" << lg.second << "\"";
      }

      std::size_t i = 0;
      for( auto el : rowEls )
      {
         if( verbose )
            std::cout << std::setw( this->widthHints[ i ] ) << el;
         if( idx++ > 0 )
            log << "," << std::endl;
         log << "         \"" << this->logsMetadata[ i ] << "\" : \"" << el << "\"";
         i++;
      }
      log << std::endl << "      }";
      this->lineStarted = true;
      if( verbose )
         std::cout << std::endl;
   }

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

   virtual void
   writeTableHeader( const std::string & spanningElement,
                     const HeaderElements & subElements ) override
   {
   }

   virtual void
   writeTableRow( const std::string & spanningElement,
                  const RowElements & subElements ) override
   {
      writeRow( subElements );
   }

   virtual void
   writeErrorMessage( const std::string& message ) override
   {
      log << "\"error\" : \"" << message << "\"" << std::endl;
   }

   virtual void
   closeTable() override
   {
   }

   virtual bool save( std::ostream & logFile ) override
   {
      if( ! this->logFileAppend )
      {
         logFile << "{" << std::endl;
         logFile << "   \"results\" : [ " << std::endl;
      }
      else
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

   // new JSON implementation
   LogsMetadata logsMetadata;
   WidthHints widthHints;
   CommonLogs commonLogs;
   std::string outputMode;

   bool lineStarted = false;
   bool resultsStarted = false;
   bool logFileAppend = false;
};

} // namespace Benchmarks
} // namespace TNL
