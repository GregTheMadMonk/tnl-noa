/***************************************************************************
                          Logger.h  -  description
                             -------------------
    begin                : 2007/08/21
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {

class Logger
{
   public:

   Logger( int _width,
              std::ostream& _stream );

   void writeHeader( const String& title );

   void writeSeparator();

   bool writeSystemInformation( const Config::ParameterContainer& parameters );
 

   void writeCurrentTime( const char* label );

   // TODO: add units
   template< typename T >
   void writeParameter( const String& label,
                        const String& parameterName,
                        const Config::ParameterContainer& parameters,
                        int parameterLevel = 0 );

   template< typename T >
   void writeParameter( const String& label,
                        const T& value,
                        int parameterLevel = 0 );

   protected:

   int width;

   std::ostream& stream;
};

} // namespace TNL

#include <TNL/Logger_impl.h>
