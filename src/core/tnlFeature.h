/***************************************************************************
                          tnlFeature.h  -  description
                             -------------------
    begin                : May 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< bool featureEnabled >
class tnlFeature
{
   public:

   enum{ enabled = featureEnabled };
};

} // namespace TNL
