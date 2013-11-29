/***************************************************************************
                          tnlDefaultSolverConrfig.h  -  description
                             -------------------
    begin                : Nov 29, 2013
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

#ifndef TNLDEFAULTSOLVERCONFIG_H_
#define TNLDEFAULTSOLVERCONFIG_H_

template< bool resolveMesh = true >
class tnlDefaultSolverConfig
{
   public:

   enum { ResolveMesh = resolveMesh };
};



#endif /* TNLDEFAULTSOLVERCONFIG_H_ */
