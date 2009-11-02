/***************************************************************************
                          mExplicitSolver.cpp  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <iomanip>
#include "mExplicitSolver.h"

//--------------------------------------------------------------------------
/*void mExplicitSolver :: SetTime( const double& t )
{
   time = t;
};
//--------------------------------------------------------------------------
const double& mExplicitSolver :: GetTime() const
{
   return time;
};
//--------------------------------------------------------------------------
void mExplicitSolver :: SetFinalTime( const double& t )
{
   final_time = t;
};
//--------------------------------------------------------------------------
long int mExplicitSolver :: GetIterationNumber() const
{
   return iteration;
};
//--------------------------------------------------------------------------
void mExplicitSolver :: SetTau( const double& t )
{
   tau = t;
};
//--------------------------------------------------------------------------
const double& mExplicitSolver :: GetTau() const
{
   return tau;
};
//--------------------------------------------------------------------------
const double& mExplicitSolver :: GetResidue() const
{
   return residue;
};
//--------------------------------------------------------------------------
void mExplicitSolver :: SetVerbosity( int v )
{
   verbosity = v;
}
//--------------------------------------------------------------------------
void mExplicitSolver :: PrintOut()
{
   if( verbosity > 0 )
   {
      // TODO: add ELA, EST, CPU
      //cout << "ELA: " << elapsed;
      //cout << " EST: " << estimated;
      //cout << " CPU: " << setw( 8 ) << user_cpu. tv_sec;
      cout << " ITER:" << setw( 9 ) << GetIterationNumber()
           << " TAU:" << setprecision( 5 ) << setw( 9 ) << GetTau()
           << " T:" << setprecision( 5 ) << setw( 9 ) << GetTime()
           << " RES:" << setprecision( 5 ) << setw( 9 ) << GetResidue();
      cout << "   \r" << flush;
   }
}*/
