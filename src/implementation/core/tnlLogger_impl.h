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

tnlLogger :: tnlLogger( int _width,
                       ostream& _stream )
: width( _width ),
  stream( _stream )
{
}

template< typename T >
void tnlLogger :: WriteParameter( const char* label,
                                  const char* parameterName,
                                  const tnlParameterContainer& parameters,
                                  int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i ++ )
      stream << " ";
   stream  << label
           << setw( width - strlen( label ) - 3 - parameterLevel )
           << parameters. GetParameter< T >( parameterName ) << " |" << endl;
}

template< typename T >
void tnlLogger :: WriteParameter( const char* label,
                                  const T& value,
                                  int parameterLevel )
{
   stream << "| ";
   int i;
   for( i = 0; i < parameterLevel; i ++ )
      stream << " ";
   stream  << label
           << setw( width - strlen( label ) - 3 - parameterLevel )
           << value << " |" << endl;
};


#endif /* TNLLOGGER_IMPL_H_ */
