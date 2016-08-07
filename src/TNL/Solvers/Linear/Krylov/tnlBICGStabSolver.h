/***************************************************************************
                          tnlBICGStabSolver.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Object.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/Vectors/SharedVector.h>
#include <TNL/Solvers/Linear/preconditioners/tnlDummyPreconditioner.h>
#include <TNL/Solvers/tnlIterativeSolver.h>
#include <TNL/Solvers/Linear/tnlLinearResidueGetter.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Krylov {

template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >

class tnlBICGStabSolver : public Object,
                          public tnlIterativeSolver< typename Matrix :: RealType,
                                                     typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;
   typedef tnlSharedPointer< MatrixType, DeviceType > MatrixPointer;
   // TODO: make this 'typedef tnlSharedPointer< const MatrixType, DeviceType > ConstMatrixPointer;'


   public:

   tnlBICGStabSolver();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setMatrix( MatrixPointer& matrix );

   void setPreconditioner( const PreconditionerType& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename VectorPointer,
             typename ResidueGetter  >
   bool solve( const VectorPointer& b, VectorPointer& x );
#else
   template< typename VectorPointer,
             typename ResidueGetter = tnlLinearResidueGetter< MatrixPointer, VectorPointer >  >
   bool solve( const VectorPointer& b, VectorPointer& x );
#endif

   ~tnlBICGStabSolver();

   protected:

   bool setSize( IndexType size );

   Vectors::Vector< RealType, DeviceType, IndexType >  r, r_ast, r_new, p, s, Ap, As, M_tmp;

   MatrixPointer matrix;
   const PreconditionerType* preconditioner;
};

} // namespace Krylov
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Krylov/tnlBICGStabSolver_impl.h>
