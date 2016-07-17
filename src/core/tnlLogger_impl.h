/***************************************************************************
                          tnlLogger_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <sstream>
#include <iomanip>

namespace TNL {

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

} // namespace TNL
