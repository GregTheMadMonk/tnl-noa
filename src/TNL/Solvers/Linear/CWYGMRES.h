/***************************************************************************
                          CWYGMRES.h  -  description
                             -------------------
    begin                : May 13, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

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
          typename Preconditioner = Preconditioners::Dummy< typename Matrix::RealType,
                                                            typename Matrix::DeviceType,
                                                            typename Matrix::IndexType> >
class CWYGMRES
: public Object,
  public IterativeSolver< typename Matrix::RealType,
                          typename Matrix::IndexType >
{
public:
   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;
   typedef SharedPointer< const MatrixType, DeviceType > MatrixPointer;
   typedef SharedPointer< const PreconditionerType, DeviceType > PreconditionerPointer;
   typedef Containers::Vector< RealType, DeviceType, IndexType > DeviceVector;
   typedef Containers::Vector< RealType, Devices::Host, IndexType > HostVector;

   CWYGMRES();

   ~CWYGMRES();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   void setRestarting( IndexType rest );

   void setMatrix( const MatrixPointer& matrix );

   void setPreconditioner( const PreconditionerPointer& preconditioner );

   template< typename Vector,
             typename ResidueGetter = LinearResidueGetter< Matrix, Vector >  >
   bool solve( const Vector& b, Vector& x );

protected:
   void hauseholder_generate( DeviceVector& Y,
                              HostVector& T,
                              const int& i,
                              DeviceVector& w );

   void hauseholder_apply_trunc( HostVector& out,
                                 DeviceVector& Y,
                                 HostVector& T,
                                 const int& i,
                                 DeviceVector& w );

   void hauseholder_cwy( DeviceVector& w,
                         DeviceVector& Y,
                         HostVector& T,
                         const int& i );

   void hauseholder_cwy_transposed( DeviceVector& w,
                                    DeviceVector& Y,
                                    HostVector& T,
                                    const int& i,
                                    DeviceVector& z );

   template< typename Vector >
   void update( IndexType k,
                IndexType m,
                const HostVector& H,
                const HostVector& s,
                DeviceVector& v,
                Vector& x );

   void generatePlaneRotation( RealType& dx,
                               RealType& dy,
                               RealType& cs,
                               RealType& sn );

   void applyPlaneRotation( RealType& dx,
                            RealType& dy,
                            RealType& cs,
                            RealType& sn );


   void setSize( IndexType _size, IndexType m );

   // single vectors
   DeviceVector r, z, w, _M_tmp;
   // matrices (in column-major format)
   DeviceVector V, Y;
   // duplicate of the upper (m+1)x(m+1) submatrix of Y (it is lower triangular) for fast access
   HostVector YL, T;
   // host-only storage for Givens rotations and the least squares problem
   HostVector cs, sn, H, s;

   IndexType size, ldSize, restarting_min, restarting_max, restarting_step_min, restarting_step_max;

   MatrixPointer matrix;
   PreconditionerPointer preconditioner;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "CWYGMRES_impl.h"
