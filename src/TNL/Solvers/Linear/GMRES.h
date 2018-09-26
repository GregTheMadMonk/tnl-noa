/***************************************************************************
                          GMRES.h  -  description
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

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class GMRES
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" ) override;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   using ConstDeviceView = Containers::VectorView< const RealType, DeviceType, IndexType >;
   using DeviceView = Containers::VectorView< RealType, DeviceType, IndexType >;
   using HostView = Containers::VectorView< RealType, Devices::Host, IndexType >;
   using DeviceVector = Containers::Vector< RealType, DeviceType, IndexType >;
   using HostVector = Containers::Vector< RealType, Devices::Host, IndexType >;

   enum class Variant { MGS, MGSR, CWY };

   int orthogonalize_MGS( const int m, const RealType normb, const RealType beta );

   int orthogonalize_CWY( const int m, const RealType normb, const RealType beta );

   void compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b );

   void preconditioned_matvec( DeviceVector& w, ConstDeviceView v );

   void hauseholder_generate( DeviceVector& Y,
                              HostVector& T,
                              const int i,
                              DeviceVector& w );

   void hauseholder_apply_trunc( HostView out,
                                 DeviceVector& Y,
                                 HostVector& T,
                                 const int i,
                                 DeviceVector& w );

   void hauseholder_cwy( DeviceVector& w,
                         DeviceVector& Y,
                         HostVector& T,
                         const int i );

   void hauseholder_cwy_transposed( DeviceVector& w,
                                    DeviceVector& Y,
                                    HostVector& T,
                                    const int i,
                                    DeviceVector& z );

   template< typename Vector >
   void update( const int k,
                const int m,
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

   void apply_givens_rotations( const int i, const int m );


   void setSize( const IndexType size );

   // selected GMRES variant
   Variant variant = Variant::CWY;

   // single vectors
   DeviceVector r, w, z, _M_tmp;
   // matrices (in column-major format)
   DeviceVector V, Y;
   // (CWY only) duplicate of the upper (m+1)x(m+1) submatrix of Y (it is lower triangular) for fast access
   HostVector YL, T;
   // host-only storage for Givens rotations and the least squares problem
   HostVector cs, sn, H, s;

   IndexType size = 0;
   IndexType ldSize = 0;
   int restarting_min = 10;
   int restarting_max = 10;
   int restarting_step_min = 3;
   int restarting_step_max = 3;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/GMRES_impl.h>
