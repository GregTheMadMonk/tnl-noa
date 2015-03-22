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

#ifndef TNLEXACTFLOWDIFFUSION_IMPL_H_
#define TNLEXACTFLOWDIFFUSION_IMPL_H_

tnlString
tnlExactFlowDiffusion< 1 >::
getType()
{
   return "tnlExactFlowDiffusion< 1 >";
}

tnlString
tnlExactFlowDiffusion< 2 >::
getType()
{
   return "tnlExactFlowDiffusion< 2 >";
}

template< typename Function, typename Vertex, typename Real >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real
tnlExactFlowDiffusion< 2 >::
getValue( const Function& function,
          const Vertex& v,
          const Real& time )
{
   return (function.template getValue< 2, 0, 0, Vertex >( v, time )*(1.0+ function.template getValue< 0, 1, 0, Vertex >( v, time )*function.template getValue< 0, 1, 0, Vertex >( v, time ))
          - 2.0*function.template getValue< 0, 1, 0, Vertex >( v, time )*function.template getValue< 1, 0, 0, Vertex >( v, time )*function.template getValue< 1, 1, 0, Vertex >( v, time )
          + function.template getValue< 0, 2, 0, Vertex >( v, time )*(1.0+ function.template getValue< 1, 0, 0, Vertex >( v, time )*function.template getValue< 1, 0, 0, Vertex >( v, time )))
          / (1.0+function.template getValue< 1, 0, 0, Vertex >( v, time )*function.template getValue< 1, 0, 0, Vertex >( v, time )
          +function.template getValue< 0, 1, 0, Vertex >( v, time )*function.template getValue< 0, 1, 0, Vertex >( v, time ));
}

tnlString
tnlExactFlowDiffusion< 3 >::
getType()
{
   return "tnlExactFlowDiffusion< 3 >";
}

#endif /* TNLEXACTFLOWDIFFUSION_IMPL_H_ */
