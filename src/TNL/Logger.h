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

   // TODO: move this to Devices::Host
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

namespace TNL {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
extern template void Logger::writeParameter< char* >( const String&,
                                                         const String&,
                                                         const Config::ParameterContainer&,
                                                         int );
extern template void Logger::writeParameter< double >( const String&,
                                                          const String&,
                                                          const Config::ParameterContainer&,
                                                          int );
extern template void Logger::writeParameter< int >( const String&,
                                                       const String&,
                                                       const Config::ParameterContainer&,
                                                       int );

// TODO: fix this
//extern template void Logger :: WriteParameter< char* >( const char*,
//                                                           const char*&,
//                                                           int );
extern template void Logger::writeParameter< double >( const String&,
                                                          const double&,
                                                          int );
extern template void Logger::writeParameter< int >( const String&,
                                                       const int&,
                                                       int );
#endif

} // namespace TNL
