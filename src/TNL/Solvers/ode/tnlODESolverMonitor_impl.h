/***************************************************************************
                          tnlODESolverMonitor_impl.h  -  description
                             -------------------
    begin                : Mar 12, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Solvers {   

template< typename RealType, typename IndexType >
tnlODESolverMonitor< RealType, IndexType> :: tnlODESolverMonitor()
: timeStep( 0.0 ),
  time( 0.0 )
{
}

template< typename RealType, typename IndexType >
void tnlODESolverMonitor< RealType, IndexType> :: refresh()
{
   if( this->verbose > 0 && this->getIterations() % this->refreshRate == 0 )
   {
      // TODO: add EST
      //cout << " EST: " << estimated;
     std::cout << " ITER:" << std::setw( 8 ) << this->getIterations()
           << " TAU:" << std::setprecision( 5 ) << std::setw( 12 ) << this->getTimeStep()
           << " T:" << std::setprecision( 5 ) << std::setw( 12 ) << this->getTime()
           << " RES:" << std::setprecision( 5 ) << std::setw( 12 ) << this->getResidue()
           << " CPU: " << std::setw( 8 ) << this->getCPUTime()
           << " ELA: " << std::setw( 8 ) << this->getRealTime();
       /*double flops = ( double ) tnl_flops_counter. getFlops();
       if( flops )
       {
        std::cout << " GFLOPS:  " << std::setw( 8 ) << 1.0e-9 * flops / rt_timer -> getTime();
       }*/
      std::cout << "   \r" << std::flush;
    }
   this->refreshing ++;
}

template< typename RealType, typename IndexType >
void tnlODESolverMonitor< RealType, IndexType> :: setTimeStep( const RealType& timeStep )
{
   this->timeStep = timeStep;
}

template< typename RealType, typename IndexType >
const RealType& tnlODESolverMonitor< RealType, IndexType> :: getTimeStep() const
{
   return this->timeStep;
}

template< typename RealType, typename IndexType >
void tnlODESolverMonitor< RealType, IndexType> :: setTime( const RealType& time )
{
   this->time = time;
}

template< typename RealType, typename IndexType >
const RealType& tnlODESolverMonitor< RealType, IndexType> :: getTime() const
{
   return this->time;
}

} // namespace Solvers
} // namespace TNL
