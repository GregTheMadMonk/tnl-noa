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
#include <TNL/solvers/preconditioners/tnlDummyPreconditioner.h>
#include <TNL/solvers/tnlIterativeSolver.h>
#include <TNL/solvers/linear/tnlLinearResidueGetter.h>

namespace TNL {

template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class tnlCGSolver : public Object,
                    public tnlIterativeSolver< typename Matrix :: RealType,
                                               typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: DeviceType Device;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;


   tnlCGSolver();
 
   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setMatrix( const MatrixType& matrix );

   void setPreconditioner( const Preconditioner& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename Vector,
             typename ResidueGetter >
   bool solve( const Vector& b, Vector& x );
#else
   template< typename Vector,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, Vector >  >
   bool solve( const Vector& b, Vector& x );
#endif

   ~tnlCGSolver();

   protected:

   bool setSize( IndexType size );

   Vectors::Vector< RealType, Device, IndexType >  r, new_r, p, Ap;

   const MatrixType* matrix;
   const PreconditionerType* preconditioner;
};

} // namespace TNL

#include <TNL/solvers/linear/krylov/tnlCGSolver_impl.h>
