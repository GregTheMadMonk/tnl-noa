/***************************************************************************
                          mLogger.h  -  description
                             -------------------
    begin                : 2007/08/21
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
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
#include "mParameterContainer.h"

class mLogger
{
   public:

   mLogger( long int _width,
            ostream& _stream )
   : width( _width ), 
     stream( _stream ){};

   void WriteHeader( const char* title );

   void WriteSeparator();

   template< typename T > void WriteParameter( const char* label,
                                               const char* parameter_name,
                                               const mParameterContainer& parameters,
                                               int parameter_level = 0 )
   {
      stream << "| ";
      int i;
      for( i = 0; i < parameter_level; i ++ )
         stream << " ";
      stream  << label 
              << setw( width - strlen( label ) - 3 - parameter_level )
              << parameters. GetParameter< T >( parameter_name ) << " |" << endl;
   };
   
   template< typename T > void WriteParameter( const char* label,
                                               const T& value,
                                               int parameter_level = 0 )
   {
      stream << "| ";
      int i;
      for( i = 0; i < parameter_level; i ++ )
         stream << " ";
      stream  << label 
              << setw( width - strlen( label ) - 3 - parameter_level )
              << value << " |" << endl;
   };


   protected:

   long int width;

   ostream& stream;
};

#endif
