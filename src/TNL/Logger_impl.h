/***************************************************************************
                          Logger_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <sstream>
#include <iomanip>

#include <TNL/Logger.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Devices/SystemInfo.h>

namespace TNL {

inline void
Logger::writeHeader( const String& title )
{
   const int fill = stream.fill();
   const int titleLength = title.getLength();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "|" << std::setw( width / 2 + titleLength / 2 )
          << title << std::setw( width / 2 - titleLength / 2  ) << "|" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream.fill( fill );
}

inline void
Logger::writeSeparator()
{
   const int fill = stream.fill();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream.fill( fill );
}

inline bool
Logger::writeSystemInformation( const Config::ParameterContainer& parameters )
{
   Devices::SystemInfo::writeDeviceInfo( *this );
   if( parameters.getParameter< String >( "device" ) == "cuda" )
      Devices::CudaDeviceInfo::writeDeviceInfo( *this );
   return true;
}

inline void
Logger::writeCurrentTime( const char* label )
{
   writeParameter< String >( label, Devices::SystemInfo::getCurrentTime() );
}

template< typename T >
void
Logger::writeParameter( const String& label,
                        const String& parameterName,
                        const Config::ParameterContainer& parameters,
                        int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i ++ )
      stream << " ";
   std::stringstream str;
   str << parameters.getParameter< T >( parameterName );
   stream << label
          << std::setw( width - label.getLength() - parameterLevel - 3 )
          << str.str() << " |" << std::endl;
}

template< typename T >
void
Logger::writeParameter( const String& label,
                        const T& value,
                        int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i ++ )
      stream << " ";
   std::stringstream str;
   str << value;
   stream << label
          << std::setw( width - label.getLength() - parameterLevel - 3 )
          << str.str() << " |" << std::endl;
}

} // namespace TNL
