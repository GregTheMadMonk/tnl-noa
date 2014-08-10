/***************************************************************************
                          tnlApproximationTest.h  -  description
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

#ifndef TNLAPPROXIMATIONTEST_H_
#define TNLAPPROXIMATIONTEST_H_

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
class tnlApproximationTest
{
   public:

      typedef typename ExactOperator::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef typename MeshType::VertexType VertexType;

      static void getEoc( const Mesh& coarserMesh,
                          const Mesh& finerMesh,
                          const RealType& refinement,
                          const ExactOperator& exactOperator,
                          const ApproximateOperator& approximateOperator,
                          const Function& function,
                          RealType& l1Eoc,
                          RealType& l2Eoc,
                          RealType& maxEoc );
};

#include <tests/approximation-tests/tnlApproximationTest_impl.h>

#endif /* TNLAPPROXIMATIONTEST_H_ */
