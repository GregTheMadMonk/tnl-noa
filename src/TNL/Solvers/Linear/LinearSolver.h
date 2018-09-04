/***************************************************************************
                          LinearSolver.h  -  description
                             -------------------
    begin                : Sep 1, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Solvers/Linear/Preconditioners/Preconditioner.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/SharedPointer.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class LinearSolver
{
public:
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorViewType = Containers::VectorView< RealType, DeviceType, IndexType >;
   using ConstVectorViewType = Containers::VectorView< typename std::add_const< RealType >::type, DeviceType, IndexType >;
   using MatrixType = Matrix;
   using MatrixPointer = SharedPointer< typename std::add_const< MatrixType >::type >;
   using PreconditionerType = Preconditioners::Preconditioner< MatrixType >;
   using PreconditionerPointer = SharedPointer< typename std::add_const< PreconditionerType >::type, DeviceType >;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" )
   {}

   virtual bool setup( const Config::ParameterContainer& parameters,
                       const String& prefix = "" )
   {
      return true;
   }

   void setMatrix( const MatrixPointer& matrix )
   {
      this->matrix = matrix;
   }

   void setPreconditioner( const PreconditionerPointer& preconditioner )
   {
      this->preconditioner = preconditioner;
   }

   virtual bool solve( ConstVectorViewType b, VectorViewType x ) = 0;

protected:
   MatrixPointer matrix;
   PreconditionerPointer preconditioner;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
