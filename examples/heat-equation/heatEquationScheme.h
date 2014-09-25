/***************************************************************************
                          heatEquationScheme.h  -  description
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


#ifndef HEATEQUATIONSCHEME_H_
#define HEATEQUATIONSCHEME_H_

template< typename Mesh,
          typename DifferentialOperator,
          typename RightHandSide >
class heatEquationScheme
{
   public:

      typedef Mesh MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef typename MeshType::VertexType VertexType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename DifferentialOperator::RealType RealType;
      typedef typename DifferentialOperator::IndexType IndexType;


      template< typename Vector >
      #ifdef HAVE_CUDA
         __device__ __host__
      #endif
         void explicitUpdate( const RealType& time,
                              const RealType& tau,
                              const MeshType& mesh,
                              const IndexType cellIndex,
                              const CoordinatesType& coordinates,
                              Vector& u,
                              Vector& fu );

   protected:

      DifferentialOperator differentialOperator;

      RightHandSide rightHandSide;

};

#include "heatEquationScheme_impl.h"

#endif /* HEATEQUATIONSCHEME_H_ */
