/***************************************************************************
                          tnlLogger.h  -  description
                             -------------------
    begin                : 2007/08/21
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <config/tnlParameterContainer.h>

namespace TNL {

class tnlLogger
{
   public:

   tnlLogger( int _width,
              ostream& _stream );

   void writeHeader( const tnlString& title );

   void writeSeparator();

   bool writeSystemInformation( const tnlParameterContainer& parameters );
 

   void writeCurrentTime( const char* label );

   // TODO: add units
   template< typename T >
   void writeParameter( const tnlString& label,
                        const tnlString& parameterName,
                        const tnlParameterContainer& parameters,
                        int parameterLevel = 0 );

   template< typename T >
   void writeParameter( const tnlString& label,
                        const T& value,
                        int parameterLevel = 0 );

   protected:

   int width;

   ostream& stream;
};

#include <core/tnlLogger_impl.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
extern template void tnlLogger::writeParameter< char* >( const tnlString&,
                                                         const tnlString&,
                                                         const tnlParameterContainer&,
                                                         int );
extern template void tnlLogger::writeParameter< double >( const tnlString&,
                                                          const tnlString&,
                                                          const tnlParameterContainer&,
                                                          int );
extern template void tnlLogger::writeParameter< int >( const tnlString&,
                                                       const tnlString&,
                                                       const tnlParameterContainer&,
                                                       int );

// TODO: fix this
//extern template void tnlLogger :: WriteParameter< char* >( const char*,
//                                                           const char*&,
//                                                           int );
extern template void tnlLogger::writeParameter< double >( const tnlString&,
                                                          const double&,
                                                          int );
extern template void tnlLogger::writeParameter< int >( const tnlString&,
                                                       const int&,
                                                       int );
#endif

} // namespace TNL
