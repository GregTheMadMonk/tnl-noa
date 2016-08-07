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
#include <TNL/Solvers/Linear/preconditioners/tnlDummyPreconditioner.h>
#include <TNL/Solvers/tnlIterativeSolver.h>
#include <TNL/Solvers/Linear/tnlLinearResidueGetter.h>
#include <TNL/tnlSharedPointer.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Krylov {   

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

} // namespace Krylov
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Krylov/tnlGMRESSolver_impl.h>

#include <TNL/Matrices/CSRMatrix.h>
#include <TNL/Matrices/EllpackMatrix.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Krylov {   
   
extern template class tnlGMRESSolver< Matrices::CSRMatrix< float,  Devices::Host, int > >;
extern template class tnlGMRESSolver< Matrices::CSRMatrix< double, Devices::Host, int > >;
extern template class tnlGMRESSolver< Matrices::CSRMatrix< float,  Devices::Host, long int > >;
extern template class tnlGMRESSolver< Matrices::CSRMatrix< double, Devices::Host, long int > >;

/*extern template class tnlGMRESSolver< EllpackMatrix< float,  Devices::Host, int > >;
extern template class tnlGMRESSolver< EllpackMatrix< double, Devices::Host, int > >;
extern template class tnlGMRESSolver< EllpackMatrix< float,  Devices::Host, long int > >;
extern template class tnlGMRESSolver< EllpackMatrix< double, Devices::Host, long int > >;

extern template class tnlGMRESSolver< MultidiagonalMatrix< float,  Devices::Host, int > >;
extern template class tnlGMRESSolver< MultidiagonalMatrix< double, Devices::Host, int > >;
extern template class tnlGMRESSolver< MultidiagonalMatrix< float,  Devices::Host, long int > >;
extern template class tnlGMRESSolver< MultidiagonalMatrix< double, Devices::Host, long int > >;*/


#ifdef HAVE_CUDA
// TODO: fix this - does not work with CUDA 5.5
/*extern template class tnlGMRESSolver< CSRMatrix< float,  Devices::Cuda, int > >;
extern template class tnlGMRESSolver< CSRMatrix< double, Devices::Cuda, int > >;
extern template class tnlGMRESSolver< CSRMatrix< float,  Devices::Cuda, long int > >;
extern template class tnlGMRESSolver< CSRMatrix< double, Devices::Cuda, long int > >;*/

/*extern template class tnlGMRESSolver< EllpackMatrix< float,  Devices::Cuda, int > >;
extern template class tnlGMRESSolver< EllpackMatrix< double, Devices::Cuda, int > >;
extern template class tnlGMRESSolver< EllpackMatrix< float,  Devices::Cuda, long int > >;
extern template class tnlGMRESSolver< EllpackMatrix< double, Devices::Cuda, long int > >;

extern template class tnlGMRESSolver< tnlMutliDiagonalMatrix< float,  Devices::Cuda, int > >;
extern template class tnlGMRESSolver< tnlMutliDiagonalMatrix< double, Devices::Cuda, int > >;
extern template class tnlGMRESSolver< tnlMutliDiagonalMatrix< float,  Devices::Cuda, long int > >;
extern template class tnlGMRESSolver< tnlMutliDiagonalMatrix< double, Devices::Cuda, long int > >;*/
#endif

} // namespace Krylov
} // namespace Linear
} // namespace Solvers
} // namespace TNL
