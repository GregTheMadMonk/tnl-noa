/***************************************************************************
                          tnlApproximationError.h  -  description
                             -------------------
    begin                : Aug 7, 2014
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

#ifndef TNLAPPROXIMATIONERROR_H_
#define TNLAPPROXIMATIONERROR_H_

#include <mesh/tnlGrid.h>

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
class tnlApproximationError
{
     public:

      typedef typename ApproximateOperator::RealType RealType;
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::VertexType VertexType;

      static void getError( const Mesh& mesh,
                            const ExactOperator& exactOperator,
                            const ApproximateOperator& approximateOperator,
                            const Function& function,
                            RealType& l1Err,
                            RealType& l2Err,
                            RealType& maxErr );
};

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
class tnlApproximationError< tnlGrid< Dimensions, Real, Device, Index >, ExactOperator, ApproximateOperator, Function >
{
   public:

      typedef typename ApproximateOperator::RealType RealType;
      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef typename MeshType::VertexType VertexType;

      static void getError( const MeshType& mesh,
                            const ExactOperator& exactOperator,
                            const ApproximateOperator& approximateOperator,
                            const Function& function,
                            RealType& l1Err,
                            RealType& l2Err,
                            RealType& maxErr );
};



#include "tnlApproximationError_impl.h"

#endif /* TNLAPPROXIMATIONERROR_H_ */
