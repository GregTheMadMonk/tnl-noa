#ifndef LaxFridrichs_H
#define LaxFridrichs_H

#include <TNL/core/vectors/tnlVector.h>
#include <TNL/mesh/tnlGrid.h>

#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsMomentum.h"
#include "LaxFridrichsEnergy.h"
#include "EulerVelGetter.h"
#include "EulerPressureGetter.h"

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichs
{
   public:
      typedef Real RealType;
      typedef Index IndexType;
      typedef tnlMeshFunction< Mesh > MeshFunctionType;
 
      typedef LaxFridrichsContinuity< Mesh, Real, Index > Continuity;
      typedef LaxFridrichsMomentum< Mesh, Real, Index > Momentum;
      typedef LaxFridrichsEnergy< Mesh, Real, Index > Energy;
      typedef EulerVelGetter< Mesh, Real, Index > Velocity;
      typedef EulerPressureGetter< Mesh, Real, Index > Pressure;
   
};

} // namespace TNL

#endif	/* LaxFridrichs_H */
