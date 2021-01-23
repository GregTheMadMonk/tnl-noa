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

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class GMRES
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
   using Traits = Linear::Traits< Matrix >;

public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   // distributed vectors/views
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;
   using VectorType = typename Traits::VectorType;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" ) override;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   // local vectors/views
   using ConstDeviceView = typename Traits::ConstLocalViewType;
   using DeviceView = typename Traits::LocalViewType;
   using DeviceVector = typename Traits::LocalVectorType;
   using HostView = typename DeviceView::template Self< RealType, Devices::Host >;
   using HostVector = typename DeviceVector::template Self< RealType, Devices::Host >;;

   enum class Variant { CGS, CGSR, MGS, MGSR, CWY };

// nvcc allows __cuda_callable__ lambdas only in public methods
#ifdef __NVCC__
public:
#endif
   int orthogonalize_CGS( const int m, const RealType normb, const RealType beta );
#ifdef __NVCC__
protected:
#endif

   int orthogonalize_MGS( const int m, const RealType normb, const RealType beta );

   int orthogonalize_CWY( const int m, const RealType normb, const RealType beta );

   void compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b );

   void preconditioned_matvec( VectorViewType w, ConstVectorViewType v );

// nvcc allows __cuda_callable__ lambdas only in public methods
#ifdef __NVCC__
public:
#endif
   void hauseholder_generate( const int i,
                              VectorViewType y_i,
                              ConstVectorViewType z );
#ifdef __NVCC__
protected:
#endif

   void hauseholder_apply_trunc( HostView out,
                                 const int i,
                                 VectorViewType y_i,
                                 ConstVectorViewType z );

   void hauseholder_cwy( VectorViewType v,
                         const int i );

// nvcc allows __cuda_callable__ lambdas only in public methods
#ifdef __NVCC__
public:
#endif
   void hauseholder_cwy_transposed( VectorViewType z,
                                    const int i,
                                    ConstVectorViewType w );
#ifdef __NVCC__
protected:
#endif

   template< typename Vector >
   void update( const int k,
                const int m,
                const HostVector& H,
                const HostVector& s,
                DeviceVector& V,
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

   void setSize( const VectorViewType& x );

   // Specialized methods to distinguish between normal and distributed matrices
   // in the implementation.
   template< typename M >
   static IndexType getLocalOffset( const M& m )
   {
      return 0;
   }

   template< typename M >
   static IndexType getLocalOffset( const Matrices::DistributedMatrix< M >& m )
   {
      return m.getLocalRowRange().getBegin();
   }

   // selected GMRES variant
   Variant variant = Variant::CWY;

   // single vectors (distributed)
   VectorType r, w, z, _M_tmp;
   // matrices (in column-major format) (local)
   DeviceVector V, Y;
   // (CWY only) duplicate of the upper (m+1)x(m+1) submatrix of Y (it is lower triangular) for fast access
   HostVector YL, T;
   // host-only storage for Givens rotations and the least squares problem
   HostVector cs, sn, H, s;

   IndexType size = 0;
   IndexType ldSize = 0;
   IndexType localOffset = 0;
   int restarting_min = 10;
   int restarting_max = 10;
   int restarting_step_min = 3;
   int restarting_step_max = 3;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/GMRES_impl.h>
