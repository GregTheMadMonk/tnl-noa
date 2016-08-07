/***************************************************************************
                          tnlCGSolver.h  -  description
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
class tnlCGSolver : public Object,
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
   typedef tnlSharedPointer< PreconditionerType, DeviceType > PreconditionerPointer;
   // TODO: make this const


   tnlCGSolver();
 
   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setMatrix( MatrixPointer& matrix );

   void setPreconditioner( const PreconditionerType& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename Vector,
             typename ResidueGetter >
   bool solve( const Vector& b, Vector& x );
#else
   template< typename VectorPointer,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, VectorPointer >  >
   bool solve( const VectorPointer& b, VectorPointer& x );
#endif

   ~tnlCGSolver();

   protected:

   bool setSize( IndexType size );

   Vectors::Vector< RealType, DeviceType, IndexType >  r, new_r, p, Ap;

   MatrixPointer matrix;
   const PreconditionerType* preconditioner;
};

} // namespace Krylov
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Krylov/tnlCGSolver_impl.h>
