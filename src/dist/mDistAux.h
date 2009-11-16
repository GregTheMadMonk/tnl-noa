/***************************************************************************
                          mDistAux.h  -  description
                             -------------------
    begin                : 2005/08/09
    copyright            : (C) 2005 by Tom� Oberhuber
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

#ifndef mDistAuxH
#define mDistAuxH

enum mDistIterMethod { mDstNone = 0,
                       mDstRegularisedScheme,
                       mDstUpwindScheme,
                       mDstGodunovScheme };

template< class T > struct mDistAux
{
   //! Coefficient for artificial viscosity method
   double epsilon;

   //! Method name 
   /*! It can be one of:
       - regularised scheme (with artificial viscosity)
       - upwind scheme
       - godunov scheme
    */
   mDistIterMethod method;

   //! Grid for gradient magnitude
   T mod_grad;

   //! Grid for laplace ( for regularised scheme )
   T laplace;

   //! Basic constructor
   mDistAux( const T& phi, 
             mDistIterMethod _method,
             const double& eps = 1.0 ) :
      epsilon( eps ),
      method( _method ),
      mod_grad( phi ),
      laplace( phi ){};
};

#endif
