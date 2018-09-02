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

#include "LinearSolver.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/Linear/Preconditioners/Dummy.h>
#include <TNL/Solvers/IterativeSolver.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix,
          typename Preconditioner = Preconditioners::Dummy< typename Matrix::RealType,
                                                            typename Matrix::DeviceType,
                                                            typename Matrix::IndexType> >
class CWYGMRES
: public LinearSolver< Matrix, Preconditioner >,
  public IterativeSolver< typename Matrix::RealType,
                          typename Matrix::IndexType >
{
   using Base = LinearSolver< Matrix, Preconditioner >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   CWYGMRES();

   ~CWYGMRES();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   void setRestarting( IndexType rest );

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   using DeviceVector = Containers::Vector< RealType, DeviceType, IndexType >;
   using HostVector = Containers::Vector< RealType, Devices::Host, IndexType >;

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
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "CWYGMRES_impl.h"
