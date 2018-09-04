/***************************************************************************
                          GMRES.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "LinearSolver.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/IterativeSolver.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class GMRES
: public LinearSolver< Matrix >,
  public IterativeSolver< typename Matrix::RealType,
                          typename Matrix::IndexType >
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

   void setRestarting( IndexType rest );

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

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

   IndexType size = 0;
   IndexType restarting_min = 10;
   IndexType restarting_max = 10;
   IndexType restarting_step_min = 3;
   IndexType restarting_step_max = 3;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/GMRES_impl.h>
