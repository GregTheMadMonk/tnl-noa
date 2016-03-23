#ifndef LaxFridrichs_H
#define LaxFridrichs_H

#include <core/vectors/tnlVector.h>
#include <mesh/tnlGrid.h>

#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsMomentum.h"
#include "LaxFridrichsEnergy.h"

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichs
{
   public:
   
      typedef LaxFridrichsContinuity< Mesh, Real, Index > Continuity;
      typedef LaxFridrichsMomentum< Mesh, Real, Index > Momentum;
      typedef LaxFridrichsEnergy< Mesh, Real, Index > Energy;
      
};

#endif	/* LaxFridrichs_H */
