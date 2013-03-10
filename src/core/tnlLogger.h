/***************************************************************************
                          tnlLogger.h  -  description
                             -------------------
    begin                : 2007/08/21
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mLoggerH
#define mLoggerH

#include <cstring>
#include <ostream>
#include <iomanip>
#include <config/tnlParameterContainer.h>

class tnlLogger
{
   public:

   tnlLogger( int _width,
              ostream& _stream );

   void WriteHeader( const tnlString& title );

   void WriteSeparator();

   bool writeSystemInformation();

   void writeCurrentTime( const char* label );

   // TODO: add units
   template< typename T > void WriteParameter( const char* label,
                                               const char* parameterName,
                                               const tnlParameterContainer& parameters,
                                               int parameterLevel = 0 );
   
   template< typename T > void WriteParameter( const char* label,
                                               const T& value,
                                               int parameterLevel = 0 );

   protected:

   int width;

   ostream& stream;
};

#include <implementation/core/tnlLogger_impl.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION
extern template void tnlLogger :: WriteParameter< char* >( const char*,
                                                           const char*,
                                                           const tnlParameterContainer&,
                                                           int );
extern template void tnlLogger :: WriteParameter< double >( const char*,
                                                            const char*,
                                                            const tnlParameterContainer&,
                                                            int );
extern template void tnlLogger :: WriteParameter< int >( const char*,
                                                         const char*,
                                                         const tnlParameterContainer&,
                                                         int );

// TODO: fix this
//extern template void tnlLogger :: WriteParameter< char* >( const char*,
//                                                           const char*&,
//                                                           int );
extern template void tnlLogger :: WriteParameter< double >( const char*,
                                                            const double&,
                                                            int );
extern template void tnlLogger :: WriteParameter< int >( const char*,
                                                         const int&,
                                                         int );
#endif

#endif
