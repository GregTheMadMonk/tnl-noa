/***************************************************************************
                          tnlODESolverMonitor_impl.h  -  description
                             -------------------
    begin                : Mar 12, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLODESOLVERMONITOR_IMPL_H_
#define TNLODESOLVERMONITOR_IMPL_H_

template< typename RealType, typename IndexType >
tnlODESolverMonitor< RealType, IndexType> :: tnlODESolverMonitor()
: timeStep( 0.0 ),
  time( 0.0 )
{
}

template< typename RealType, typename IndexType >
void tnlODESolverMonitor< RealType, IndexType> :: refresh( bool force )
{
   if(  this->verbose > 0 && ( force || this->getIterations() % this->refreshRate == 0 ) )
   {
      // TODO: add EST
      //cout << " EST: " << estimated;
      cout << " ITER:" << setw( 8 ) << this->getIterations()
           << " TAU:" << setprecision( 5 ) << setw( 12 ) << this->getTimeStep()
           << " T:" << setprecision( 5 ) << setw( 12 ) << this->getTime()
           << " RES:" << setprecision( 5 ) << setw( 12 ) << this->getResidue()
           << " CPU: " << setw( 8 ) << this->getCPUTime()
           << " ELA: " << setw( 8 ) << this->getRealTime();
       /*double flops = ( double ) tnl_flops_counter. getFlops();
       if( flops )
       {
         cout << " GFLOPS:  " << setw( 8 ) << 1.0e-9 * flops / rt_timer -> getTime();
       }*/
       cout << "   \r" << flush;
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

#endif /* TNLODESOLVERMONITOR_IMPL_H_ */
