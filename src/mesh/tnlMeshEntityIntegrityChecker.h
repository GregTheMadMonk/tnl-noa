/***************************************************************************
                          tnlMeshEntityIntegrityChecker.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
