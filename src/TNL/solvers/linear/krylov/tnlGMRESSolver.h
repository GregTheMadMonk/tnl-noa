/***************************************************************************
                          tnlGMRESSolver.h  -  description
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
#include <TNL/tnlSharedPointer.h>

namespace TNL {

template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class tnlGMRESSolver : public Object,
                       public tnlIterativeSolver< typename Matrix :: RealType,
                                                  typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;
   typedef tnlSharedPointer< MatrixType, DeviceType > MatrixPointer;
   // TODO: make this: typedef tnlSharedPointer< const MatrixType, DeviceType > ConstMatrixPointer;

   tnlGMRESSolver();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setRestarting( IndexType rest );

   void setMatrix( MatrixPointer& matrix );

   void setPreconditioner( const PreconditionerType& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename VectorPointer,
             typename ResidueGetter >
   bool solve( const Vector& b, Vector& x );
#else
   template< typename VectorPointer,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, VectorPointer >  >
   bool solve( const VectorPointer& b, VectorPointer& x );
#endif

   ~tnlGMRESSolver();

   protected:

   template< typename VectorT >
   void update( IndexType k,
                IndexType m,
                const Vectors::Vector< RealType, Devices::Host, IndexType >& H,
                const Vectors::Vector< RealType, Devices::Host, IndexType >& s,
                Vectors::Vector< RealType, DeviceType, IndexType >& v,
                VectorT& x );

   void generatePlaneRotation( RealType &dx,
                               RealType &dy,
                               RealType &cs,
                               RealType &sn );

   void applyPlaneRotation( RealType &dx,
                            RealType &dy,
                            RealType &cs,
                            RealType &sn );


   bool setSize( IndexType _size, IndexType m );

   Vectors::Vector< RealType, DeviceType, IndexType > _r, w, _v, _M_tmp;
   Vectors::Vector< RealType, Devices::Host, IndexType > _s, _cs, _sn, _H;

   IndexType size, restarting;

   MatrixPointer matrix;
   const PreconditionerType* preconditioner;
};

} // namespace TNL

#include <TNL/solvers/linear/krylov/tnlGMRESSolver_impl.h>

#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/matrices/tnlEllpackMatrix.h>
#include <TNL/matrices/tnlMultidiagonalMatrix.h>

namespace TNL {

extern template class tnlGMRESSolver< tnlCSRMatrix< float,  Devices::Host, int > >;
extern template class tnlGMRESSolver< tnlCSRMatrix< double, Devices::Host, int > >;
extern template class tnlGMRESSolver< tnlCSRMatrix< float,  Devices::Host, long int > >;
extern template class tnlGMRESSolver< tnlCSRMatrix< double, Devices::Host, long int > >;

/*extern template class tnlGMRESSolver< tnlEllpackMatrix< float,  Devices::Host, int > >;
extern template class tnlGMRESSolver< tnlEllpackMatrix< double, Devices::Host, int > >;
extern template class tnlGMRESSolver< tnlEllpackMatrix< float,  Devices::Host, long int > >;
extern template class tnlGMRESSolver< tnlEllpackMatrix< double, Devices::Host, long int > >;

extern template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  Devices::Host, int > >;
extern template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, Devices::Host, int > >;
extern template class tnlGMRESSolver< tnlMultiDiagonalMatrix< float,  Devices::Host, long int > >;
extern template class tnlGMRESSolver< tnlMultiDiagonalMatrix< double, Devices::Host, long int > >;*/


#ifdef HAVE_CUDA
// TODO: fix this - does not work with CUDA 5.5
/*extern template class tnlGMRESSolver< tnlCSRMatrix< float,  Devices::Cuda, int > >;
extern template class tnlGMRESSolver< tnlCSRMatrix< double, Devices::Cuda, int > >;
extern template class tnlGMRESSolver< tnlCSRMatrix< float,  Devices::Cuda, long int > >;
extern template class tnlGMRESSolver< tnlCSRMatrix< double, Devices::Cuda, long int > >;*/

/*extern template class tnlGMRESSolver< tnlEllpackMatrix< float,  Devices::Cuda, int > >;
extern template class tnlGMRESSolver< tnlEllpackMatrix< double, Devices::Cuda, int > >;
extern template class tnlGMRESSolver< tnlEllpackMatrix< float,  Devices::Cuda, long int > >;
extern template class tnlGMRESSolver< tnlEllpackMatrix< double, Devices::Cuda, long int > >;

extern template class tnlGMRESSolver< tnlMutliDiagonalMatrix< float,  Devices::Cuda, int > >;
extern template class tnlGMRESSolver< tnlMutliDiagonalMatrix< double, Devices::Cuda, int > >;
extern template class tnlGMRESSolver< tnlMutliDiagonalMatrix< float,  Devices::Cuda, long int > >;
extern template class tnlGMRESSolver< tnlMutliDiagonalMatrix< double, Devices::Cuda, long int > >;*/
#endif

} // namespace TNL
