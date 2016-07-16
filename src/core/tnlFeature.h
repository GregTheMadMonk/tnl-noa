/***************************************************************************
                          tnlFeature.h  -  description
                             -------------------
    begin                : May 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLFEATURE_H_
#define TNLFEATURE_H_

template< bool featureEnabled >
class tnlFeature
{
   public:

   enum{ enabled = featureEnabled };
};


#endif /* TNLFEATURE_H_ */
