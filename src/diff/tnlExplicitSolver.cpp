/***************************************************************************
                          tnlExplicitSolver.cpp  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomas Oberhuber
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

#include <iomanip>
#include "tnlExplicitSolver.h"

//--------------------------------------------------------------------------
/*void tnlExplicitSolver :: SetTime( const double& t )
{
   time = t;
};
//--------------------------------------------------------------------------
const double& tnlExplicitSolver :: GetTime() const
{
   return time;
};
//--------------------------------------------------------------------------
void tnlExplicitSolver :: SetFinalTime( const double& t )
{
   final_time = t;
};
//--------------------------------------------------------------------------
int tnlExplicitSolver :: GetIterationNumber() const
{
   return iteration;
};
//--------------------------------------------------------------------------
void tnlExplicitSolver :: SetTau( const double& t )
{
   tau = t;
};
//--------------------------------------------------------------------------
const double& tnlExplicitSolver :: GetTau() const
{
   return tau;
};
//--------------------------------------------------------------------------
const double& tnlExplicitSolver :: GetResidue() const
{
   return residue;
};
//--------------------------------------------------------------------------
void tnlExplicitSolver :: SetVerbosity( int v )
{
   verbosity = v;
}
//--------------------------------------------------------------------------
void tnlExplicitSolver :: PrintOut()
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
