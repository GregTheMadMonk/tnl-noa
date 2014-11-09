/***************************************************************************
                          tnlSDFSchemeTest_impl.h  -  description
                             -------------------
    begin                : Nov 19, 2013
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

#ifndef TNLSDFSCHEMETEST_IMPL_H_
#define TNLSDFSCHEMETEST_IMPL_H_

#include <functions/tnlSDFSchemeTest.h>

template< typename function, typename Real >
tnlSDFSchemeTestBase< function, Real >::tnlSDFSchemeTestBase()
{
}

template< typename function, typename Real >
bool tnlSDFSchemeTestBase< function, Real >::v( const tnlParameterContainer& parameters,
        const tnlString& prefix = "" )
{
	f.init(parameters);

   return true;
}



template< typename function, int Dimensions, typename Real >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
Real tnlSDFSchemeTest< function, 1, Real >::getValue( const Vertex& v,
              const Real& time = 0.0 ) const
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 || XDiffOrder != 0 )
      return 0.0;

   return Sign( this->f.getValue<0,0,0>(v))*
		   	   ( 1-sqrt(this->f.getValue<1,0,0>(v)*this->f.getValue<1,0,0>(v)) );
}


template< typename function, int Dimensions, typename Real >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
Real tnlSDFSchemeTest< function, 2, Real >::getValue( const Vertex& v,
              const Real& time = 0.0 ) const
{
	   if( YDiffOrder != 0 || ZDiffOrder != 0 || XDiffOrder != 0 )
	      return 0.0;

	   return Sign( this->f.getValue<0,0,0>(v))*
			   ( 1-sqrt(this->f.getValue<1,0,0>(v)*this->f.getValue<1,0,0>(v) +
					    this->f.getValue<0,1,0>(v)*this->f.getValue<0,1,0>(v)) );
}

template< typename function, int Dimensions, typename Real >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
Real tnlSDFSchemeTest< function, 3, Real >::getValue( const Vertex& v,
              const Real& time = 0.0 ) const
{
	   if( YDiffOrder != 0 || ZDiffOrder != 0 || XDiffOrder != 0 )
	      return 0.0;

	   return Sign( this->f.getValue<0,0,0>(v))*
			   ( 1.0-sqrt(this->f.getValue<1,0,0>(v)*this->f.getValue<1,0,0>(v) +
					      this->f.getValue<0,1,0>(v)*this->f.getValue<0,1,0>(v) +
					      this->f.getValue<0,0,1>(v)*this->f.getValue<0,0,1>(v)) );
}

#endif /* TNLSDFSCHEMETEST_IMPL_H_ */
