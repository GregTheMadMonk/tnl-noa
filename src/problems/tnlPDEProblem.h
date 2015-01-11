/***************************************************************************
                          tnlPDEProblem.h  -  description
                             -------------------
    begin                : Jan 10, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLPDEPROBLEM_H_
#define TNLPDEPROBLEM_H_

#include<problems/tnlProblem.h>
#include<matrices/tnlCSRMatrix.h>

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Device = typename Mesh::DeviceType,
          typename Index = typename Mesh::IndexType >
class tnlPDEProblem : public tnlProblem< Real, Device, Index >
{
   public:

      typedef tnlProblem< Real, Device, Index > BaseType;
      using typename BaseType::RealType;
      using typename BaseType::DeviceType;
      using typename BaseType::IndexType;

      typedef Mesh MeshType;
      typedef tnlVector< RealType, DeviceType, IndexType> DofVectorType;
      typedef tnlCSRMatrix< RealType, DeviceType, IndexType > MatrixType;

      static tnlString getTypeStatic();

      tnlString getPrologHeader() const;

      void writeProlog( tnlLogger& logger,
                        const tnlParameterContainer& parameters ) const;

      IndexType getAuxiliaryDofs( const MeshType& mesh ) const;

      void bindAuxiliaryDofs( const MeshType& mesh,
                              DofVectorType& auxiliaryDofs );

      bool preIterate( const RealType& time,
                       const RealType& tau,
                       const MeshType& mesh,
                       DofVectorType& dofs,
                       DofVectorType& auxDofs );

      bool postIterate( const RealType& time,
                        const RealType& tau,
                        const MeshType& mesh,
                        DofVectorType& dofs,
                        DofVectorType& auxDofs );


      tnlSolverMonitor< RealType, IndexType >* getSolverMonitor();


};

#include <problems/tnlPDEProblem_impl.h>

#endif /* TNLPDEPROBLEM_H_ */
