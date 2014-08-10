/***************************************************************************
                          tnlExactLinearDiffusion_impl.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLEXACTLINEARDIFFUSION_IMPL_H_
#define TNLEXACTLINEARDIFFUSION_IMPL_H_

template<>
   template< typename Function, typename Vertex >
typename Function::RealType
tnlExactLinearDiffusion< 1 >::
getValue( const Function& function,
          const Vertex& v )
{
   return function.template getValue< 2 >( v );
}

template<>
   template< typename Function, typename Vertex >
typename Function::RealType
tnlExactLinearDiffusion< 2 >::
getValue( const Function& function,
          const Vertex& v )
{
   return function.template getValue< 2, 0 >( v ) +
          function.template getValue< 0, 2 >( v );
}

template<>
   template< typename Function, typename Vertex >
typename Function::RealType
tnlExactLinearDiffusion< 3 >::
getValue( const Function& function,
          const Vertex& v )
{
   return function.template getValue< 2, 0, 0 >( v ) +
          function.template getValue< 0, 2, 0 >( v ) +
          function.template getValue< 0, 0, 2 >( v );

}

#endif /* TNLEXACTLINEARDIFFUSION_IMPL_H_ */
