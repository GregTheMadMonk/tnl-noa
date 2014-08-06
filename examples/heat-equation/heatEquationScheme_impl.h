/***************************************************************************
                          heatEquationScheme_impl.h  -  description
                             -------------------
    begin                : Aug 05, 2014
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

#ifndef HEATEQUATIONSCHEME_IMPL_H_
#define HEATEQUATIONSCHEME_IMPL_H_

template< typename Mesh,
          typename DifferentialOperator,
          typename RightHandSide >
   template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void
heatEquationScheme< Mesh, DifferentialOperator, RightHandSide >::
explicitUpdate( const RealType& time,
                const RealType& tau,
                const MeshType& mesh,
                const IndexType cellIndex,
                const CoordinatesType& coordinates,
                Vector& u,
                Vector& fu )
{
   this->differentialOperator.explicitUpdate( time, tau, mesh, cellIndex, coordinates, u, fu );
   VertexType vertex = mesh.getCellCenter( coordinates );
   fu[ cellIndex ] += this->rightHandSide.getValue( vertex );
}

#endif /* HEATEQUATIONSCHEME_IMPL_H_ */
