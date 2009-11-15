/***************************************************************************
                          msdfSussmanFatemi.h  -  description
                             -------------------
    begin                : 2008/03/13
    copyright            : (C) 2008 by Tomá¹ Oberhuber
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

#ifndef msdfSussmanFatemiH
#define msdfSussmanFatemiH

#include <diff/mdiff.h>
#include "msdfIterBase.h"

class msdfSussmanFatemi : public msdfIterBase
{
   public:
   
   msdfSussmanFatemi();

   bool Init( const mParameterContainer& parameters,
              mGrid2D< double >* u );

   void SetTimeDiscretisation( const char* str = 0 );

   bool SetSolver( const mGrid2D< double >* u, const char* str = 0 );

   void SetInitialCondition( mGrid2D< double >& u_ini );

   void SetOutputPeriod( const double& t );

   void SetTau( const double& t );
   
   void SetFinalTime( const double& t );

   void SetMersonAdaptivity( const double& t );

   void SetVerbosity( const int );

   void GetExplicitRHS( const double&, //time
                        mGrid2D< double >&, //_u
                        mGrid2D< double >& ); // _fu
   
   ~msdfSussmanFatemi();

   private:

   mGrid2D< double > *_u0, *_heaviside_prime, *_l, *_mod_grad_u0, *_sign_u0, *_g2;
   
};

#endif
