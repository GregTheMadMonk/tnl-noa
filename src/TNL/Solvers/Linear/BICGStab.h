/***************************************************************************
                          BICGStab.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Object.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Containers/Vector.h>
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

class BICGStab : public Object,
                 public IterativeSolver< typename Matrix :: RealType,
                                         typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;
   typedef SharedPointer< const MatrixType, DeviceType > MatrixPointer;
   typedef SharedPointer< const PreconditionerType, DeviceType > PreconditionerPointer;

   BICGStab();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   void setMatrix( const MatrixPointer& matrix );

   void setPreconditioner( const PreconditionerPointer& preconditioner );

   template< typename Vector,
             typename ResidueGetter = LinearResidueGetter< Matrix, Vector >  >
   bool solve( const Vector& b, Vector& x );

   protected:

   void setSize( IndexType size );

   bool exact_residue;

   Containers::Vector< RealType, DeviceType, IndexType > r, r_ast, p, s, Ap, As, M_tmp;

   MatrixPointer matrix;
   PreconditionerPointer preconditioner;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/BICGStab_impl.h>
