/***************************************************************************
                          tnlOmp.h  -  description
                             -------------------
    begin                : Mar 4, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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


#ifndef TNLOMP_H
#define	TNLOMP_H

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

class tnlOmp
{
   public:      
      
      static void disable();
      
      static void enable();
      
      static inline bool isEnabled() { return enabled; };
      
      static void setMaxThreadsCount( int maxThreadsCount );
      
      static int getMaxThreadsCount();
      
      static int getThreadIdx();
      
      static void configSetup( tnlConfigDescription& config, const tnlString& prefix = "" );
      
      static bool setup( const tnlParameterContainer& parameters,
                         const tnlString& prefix = "" );
            
   protected:
      
      static bool enabled;
      
      static int maxThreadsCount;
};


#endif	/* TNLOMP_H */

