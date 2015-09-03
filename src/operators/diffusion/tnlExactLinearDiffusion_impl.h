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

tnlString
tnlExactLinearDiffusion< 1 >::
getType()
{
   return "tnlExactLinearDiffusion< 1 >";
}

template< typename Function, typename Vertex, typename Real >
__cuda_callable__
Real
tnlExactLinearDiffusion< 1 >::
getValue( const Function& function,
          const Vertex& v,
          const Real& time )
{
   return function.template getValue< 2, 0, 0, Vertex >( v, time );
}

tnlString
tnlExactLinearDiffusion< 2 >::
getType()
{
   return "tnlExactLinearDiffusion< 2 >";
}

template< typename Function, typename Vertex, typename Real >
__cuda_callable__
Real
tnlExactLinearDiffusion< 2 >::
getValue( const Function& function,
          const Vertex& v,
          const Real& time )
{
   return function.template getValue< 2, 0, 0, Vertex >( v, time ) +
          function.template getValue< 0, 2, 0, Vertex >( v, time );
}

tnlString
tnlExactLinearDiffusion< 3 >::
getType()
{
   return "tnlExactLinearDiffusion< 3 >";
}

template< typename Function, typename Vertex, typename Real >
__cuda_callable__
Real
tnlExactLinearDiffusion< 3 >::
getValue( const Function& function,
          const Vertex& v,
          const Real& time )
{
   return function.template getValue< 2, 0, 0, Vertex >( v, time ) +
          function.template getValue< 0, 2, 0, Vertex >( v, time ) +
          function.template getValue< 0, 0, 2, Vertex >( v, time );

}

#endif /* TNLEXACTLINEARDIFFUSION_IMPL_H_ */
