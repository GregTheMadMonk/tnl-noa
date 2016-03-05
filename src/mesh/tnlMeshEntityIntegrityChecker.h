/***************************************************************************
                          tnlMeshEntityIntegrityChecker.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLMESHENTITYINTEGRITYCHECKER_H_
#define TNLMESHENTITYINTEGRITYCHECKER_H_

template< typename MeshEntity >
class tnlMeshEntityIntegrityChecker
{
   public:

      static bool checkEntity( const MeshEntity& entity )
      {
         return true;
      }

};


#endif /* TNLMESHENTITYINTEGRITYCHECKER_H_ */
