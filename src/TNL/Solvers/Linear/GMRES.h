/***************************************************************************
                          GMRES.h  -  description
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
class GMRES
: public Object,
  public IterativeSolver< typename Matrix :: RealType,
                          typename Matrix :: IndexType >
{
public:
   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;
   typedef Pointers::SharedPointer<  const MatrixType, DeviceType > MatrixPointer;
   typedef Pointers::SharedPointer<  const PreconditionerType, DeviceType > PreconditionerPointer;

   GMRES();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   void setRestarting( IndexType rest );

   void setMatrix( const MatrixPointer& matrix );

   void setPreconditioner( const PreconditionerPointer& preconditioner );

   template< typename Vector,
             typename ResidueGetter = LinearResidueGetter< Matrix, Vector > >
   bool solve( const Vector& b, Vector& x );

   ~GMRES();

protected:
   template< typename VectorT >
   void update( IndexType k,
                IndexType m,
                const Containers::Vector< RealType, Devices::Host, IndexType >& H,
                const Containers::Vector< RealType, Devices::Host, IndexType >& s,
                Containers::Vector< RealType, DeviceType, IndexType >& v,
                VectorT& x );

   void generatePlaneRotation( RealType& dx,
                               RealType& dy,
                               RealType& cs,
                               RealType& sn );

   void applyPlaneRotation( RealType& dx,
                            RealType& dy,
                            RealType& cs,
                            RealType& sn );


   void setSize( IndexType _size, IndexType m );

   Containers::Vector< RealType, DeviceType, IndexType > _r, w, _v, _M_tmp;
   Containers::Vector< RealType, Devices::Host, IndexType > _s, _cs, _sn, _H;

   IndexType size, restarting_min, restarting_max, restarting_step_min, restarting_step_max;

   MatrixPointer matrix;
   
   PreconditionerPointer preconditioner;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/GMRES_impl.h>

#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/Multidiagonal.h>

namespace TNL {
namespace Solvers {
namespace Linear {
   
/*extern template class GMRES< Matrices::CSR< float,  Devices::Host, int > >;
extern template class GMRES< Matrices::CSR< double, Devices::Host, int > >;
extern template class GMRES< Matrices::CSR< float,  Devices::Host, long int > >;
extern template class GMRES< Matrices::CSR< double, Devices::Host, long int > >;*/

/*extern template class GMRES< Ellpack< float,  Devices::Host, int > >;
extern template class GMRES< Ellpack< double, Devices::Host, int > >;
extern template class GMRES< Ellpack< float,  Devices::Host, long int > >;
extern template class GMRES< Ellpack< double, Devices::Host, long int > >;

extern template class GMRES< Multidiagonal< float,  Devices::Host, int > >;
extern template class GMRES< Multidiagonal< double, Devices::Host, int > >;
extern template class GMRES< Multidiagonal< float,  Devices::Host, long int > >;
extern template class GMRES< Multidiagonal< double, Devices::Host, long int > >;*/


#ifdef HAVE_CUDA
// TODO: fix this - does not work with CUDA 5.5
/*extern template class GMRES< CSR< float,  Devices::Cuda, int > >;
extern template class GMRES< CSR< double, Devices::Cuda, int > >;
extern template class GMRES< CSR< float,  Devices::Cuda, long int > >;
extern template class GMRES< CSR< double, Devices::Cuda, long int > >;*/

/*extern template class GMRES< Ellpack< float,  Devices::Cuda, int > >;
extern template class GMRES< Ellpack< double, Devices::Cuda, int > >;
extern template class GMRES< Ellpack< float,  Devices::Cuda, long int > >;
extern template class GMRES< Ellpack< double, Devices::Cuda, long int > >;

extern template class GMRES< tnlMutliDiagonalMatrix< float,  Devices::Cuda, int > >;
extern template class GMRES< tnlMutliDiagonalMatrix< double, Devices::Cuda, int > >;
extern template class GMRES< tnlMutliDiagonalMatrix< float,  Devices::Cuda, long int > >;
extern template class GMRES< tnlMutliDiagonalMatrix< double, Devices::Cuda, long int > >;*/
#endif

} // namespace Linear
} // namespace Solvers
} // namespace TNL
