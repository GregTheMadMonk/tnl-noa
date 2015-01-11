/***************************************************************************
                          tnlLogger_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLLOGGER_IMPL_H_
#define TNLLOGGER_IMPL_H_

#include <sstream>

template< typename T >
void tnlLogger::writeParameter( const tnlString& label,
                                const tnlString& parameterName,
                                const tnlParameterContainer& parameters,
                                int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i ++ )
      stream << " ";
   std::stringstream str;
   str << parameters.getParameter< T >( parameterName );
   stream  << label
           << setw( width - label.getLength() - parameterLevel - 3 )
           << str.str() << " |" << endl;
}

template< typename T >
void tnlLogger :: writeParameter( const tnlString& label,
                                  const T& value,
                                  int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i ++ )
      stream << " ";
   std::stringstream str;
   str << value;
   stream  << label
           << setw( width - label.getLength() - parameterLevel - 3 )
           << str.str() << " |" << endl;
};

#endif /* TNLLOGGER_IMPL_H_ */
