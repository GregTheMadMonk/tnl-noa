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

   /////
   /// \brief Basic constructor.
   ///
   /// \param _width Integer that defines the width of logger.
   /// \param _stream Where to create the logger, e.g. cout or a certain files.
   Logger( int _width,
              std::ostream& _stream );

   /////
   /// \brief Creates header in given logger.
   ///
   /// \param title String desribing the title/header.
   void writeHeader( const String& title );

   /// \brief Creates predefined separator - structure in the logger.
   void writeSeparator();

   /// \brief Inserts information about various system parameters into logger.
   ///
   /// \param parameters
   bool writeSystemInformation( const Config::ParameterContainer& parameters );

   /////
   /// \brief Inserts a line with current time into logger.
   ///
   /// \param label Description of the current time line.
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
