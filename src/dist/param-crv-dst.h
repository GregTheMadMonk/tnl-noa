/***************************************************************************
                          param-crv-dst.h  -  description
                             -------------------
    begin                : 2007/02/24
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

#ifndef param_crv_dstH
#define param_crv_dstH

#include <diff/mGrid2D.h>

class mLevelSetCreator
{
   mField2D< bool >* fixed_points;

   public:

   mLevelSetCreator();

   ~mLevelSetCreator();

   bool Init( mGrid2D< double >& u );

   void DrawCurve( mGrid2D< double >& u,
                   void ( *crv )( const double& t, void* crv_data, double& pos_x, double& pos_y ),
                   void* crv_data,
                   const double& t1,
                   const double& t2,
                   const double quality = 2.0 );

   void Finalize( mGrid2D< double >& u,
                  const int sweepings = 2 );
};



void GetParametrisedCurveSDF( mGrid2D< double >& u, 
                              void ( *crv )( const double& t, void* crv_data, double& pos_x, double& pos_y ),
                              void* crv_data,
                              const double& t1,
                              const double& t2,
                              const int sweepings = 24,
                              const double quality = 2.0 );

#endif
