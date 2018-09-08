/***************************************************************************
                          StaticVectorFor.h  -  description
                             -------------------
    begin                : July 12, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/StaticVector.h>

namespace TNL {

struct StaticVectorFor
{
    template < typename Index,
             typename Function,
             typename... FunctionArgs,
             int dim>
    static void exec( Containers::StaticVector<dim,Index> starts, Containers::StaticVector<dim,Index> ends, Function f, FunctionArgs... args )
    {
        Containers::StaticVector<dim,Index> index;
        if(dim==1)
        {
            for(index[0]=starts[0]; index[0]< ends[0];index[0]++ )
                 f( index, args... );
        }

        if(dim==2)
        {
            for(index[1]=starts[1]; index[1]< ends[1];index[1]++ )
                for(index[0]=starts[0]; index[0]< ends[0];index[0]++ )
                        f( index, args... );
        }

        if(dim==3)
        {
            for(index[2]=starts[2]; index[2]< ends[2];index[2]++ )
                for(index[1]=starts[1]; index[1]< ends[1];index[1]++ )
                    for(index[0]=starts[0]; index[0]< ends[0];index[0]++ )
                        f( index, args... );
        }
    }
};

} // namespace TNL
