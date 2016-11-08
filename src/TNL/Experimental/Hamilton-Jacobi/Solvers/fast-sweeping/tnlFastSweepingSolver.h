/* 
 * File:   tnlFastSweepingSolver.h
 * Author: oberhuber
 *
 * Created on July 12, 2016, 6:04 PM
 */

#pragma once

#include <functions/tnlConstantFunction.h>
#include <problems/tnlPDEProblem.h>

template< typename Mesh,
          typename Anisotropy = tnlConstanstFunction< Mesh > >
class tnlFastSweepingSolver  : public tnlPDEProblem< Mesh,
                                                     typename Mesh::RealType,
                                                     typename Mesh::DeviceType,
                                                     typename Mesh::IndexType  >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;

      typedef tnlMeshFunction< Mesh > MeshFunctionType;
      typedef tnlPDEProblem< Mesh, TimeDependentProblem, RealType, DeviceType, IndexType > BaseType;

      using typename BaseType::MeshType;
      using typename BaseType::DofVectorType;
      using typename BaseType::MeshDependentDataType;
};


