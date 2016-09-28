/***************************************************************************
                          TFQMR.h  -  description
                             -------------------
    begin                : Dec 8, 2012
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Object.h>
#include <TNL/SharedPointer.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/Solvers/Linear/Preconditioners/Dummy.h>
#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Solvers/Linear/LinearResidueGetter.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix,
          typename Preconditioner = Preconditioners::Dummy< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >

class TFQMR : public Object,
              public IterativeSolver< typename Matrix :: RealType,
                                      typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;
   typedef SharedPointer< const MatrixType, DeviceType, true > MatrixPointer;
   typedef SharedPointer< const PreconditionerType, DeviceType, true > PreconditionerPointer;

   TFQMR();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   void setMatrix( const MatrixPointer& matrix );

   void setPreconditioner( const PreconditionerPointer& preconditioner );

   template< typename VectorPointer,
             typename ResidueGetter = LinearResidueGetter< Matrix, typename VectorPointer::ObjectType >  >
   bool solve( const VectorPointer& b, VectorPointer& x );

   ~TFQMR();

   protected:

   bool setSize( IndexType size );

   Containers::Vector< RealType, DeviceType, IndexType >  d, r, w, u, v, r_ast, Au, M_tmp;

   IndexType size;

   MatrixPointer matrix;
   PreconditionerPointer preconditioner;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/TFQMR_impl.h>
