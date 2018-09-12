/***************************************************************************
                          Logger.cpp  -  description
                             -------------------
    begin                : 2007/08/22
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <iomanip>
#include <TNL/Logger.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Devices/SystemInfo.h>

namespace TNL {

Logger :: Logger( int _width, std::ostream& _stream )
: width( _width ),
  stream( _stream )
{
}

void Logger :: writeHeader( const String& title )
{
   int fill = stream. fill();
   int titleLength = title. getLength();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "|" << std::setw( width / 2 + titleLength / 2 )
    << title << std::setw( width / 2 - titleLength / 2  ) << "|" << std::endl;
   stream << "|" << std::setfill( ' ' ) << std::setw( width ) << "|" << std::endl;
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream. fill( fill );
}

void Logger :: writeSeparator()
{
   int fill = stream. fill();
   stream << "+" << std::setfill( '-' ) << std::setw( width ) << "+" << std::endl;
   stream. fill( fill );
}

bool Logger :: writeSystemInformation( const Config::ParameterContainer& parameters )
{
   Devices::SystemInfo::writeDeviceInfo( *this );
   if( parameters.getParameter< String >( "device" ) == "cuda" )
      Devices::CudaDeviceInfo::writeDeviceInfo( *this );
   return true;
}

void Logger :: writeCurrentTime( const char* label )
{
   writeParameter< String >( label, Devices::SystemInfo::getCurrentTime() );
}

} // namespace TNL
