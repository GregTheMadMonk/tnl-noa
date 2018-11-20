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

/// Vytvari tabulku s logem vypoctu   
class Logger
{
   public:

   /////
   /// \brief Basic constructor.
   ///
   /// \param _width Integer that defines the width of the log.
   /// \param _stream Defines output stream where the log will be printed out.
   Logger( int _width,
              std::ostream& _stream );

   /////
   /// \brief Creates header in given log.
   /// 
   /// The header usually contains title of the program.
   ///
   /// \param title String containing the header title.
   void writeHeader( const String& title );

   /// \brief Creates separator for structuring the log.
   void writeSeparator();

   /// \brief Inserts information about various system parameters into the log.
   ///
   /// \param parameters is a container with configuration parameters
   bool writeSystemInformation( const Config::ParameterContainer& parameters );

   /////
   /// \brief Inserts a line with current time into the log.
   ///
   /// \param label Label to be printed to the log together with the current time.
   void writeCurrentTime( const char* label );

   // TODO: add units
   template< typename ParameterType >
   void writeParameter( const String& label,
                        const String& parameterName,
                        const Config::ParameterContainer& parameters,
                        int parameterLevel = 0 );

   template< typename ParameterType >
   void writeParameter( const String& label,
                        const ParameterType& value,
                        int parameterLevel = 0 );

   protected:

   int width;

   std::ostream& stream;
};

} // namespace TNL

#include <TNL/Logger_impl.h>
