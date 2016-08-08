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
#include <TNL/Vectors/Vector.h>
#include <TNL/Vectors/SharedVector.h>
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
   typedef tnlSharedPointer< MatrixType, DeviceType > MatrixPointer;
   // TODO: make this: typedef tnlSharedPointer< const MatrixType, DeviceType > ConstMatrixPointer;

   public:

   TFQMR();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setMatrix( MatrixPointer& matrix );

   void setPreconditioner( const Preconditioner& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename VectorPointer,
             typename ResidueGetter >
   bool solve( const VectorPointer& b, VectorPointer& x );
#else
   template< typename VectorPointer,
             typename ResidueGetter = LinearResidueGetter< Matrix, VectorPointer >  >
   bool solve( const VectorPointer& b, VectorPointer& x );
#endif

   ~TFQMR();

   protected:

   bool setSize( IndexType size );

   Vectors::Vector< RealType, DeviceType, IndexType >  d, r, w, u, v, r_ast, Au, M_tmp;

   IndexType size;

   MatrixPointer matrix;
   const PreconditionerType* preconditioner;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/TFQMR_impl.h>
