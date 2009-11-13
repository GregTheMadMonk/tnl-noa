/***************************************************************************
                          polygonizer.h  -  description
                             -------------------
    begin                : Mon Feb 11 2002
    copyright            : (C) 2002 by Tomá¹ Oberhuber
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

#ifndef POLYGONIZER_H
#define POLYGONIZER_H


/**
  *@author Tomá¹ Oberhuber
  *
  * this class is rewritten polygonizer by Jules Bloomenthal, Xerox PARC.
  * Copyright of original code (c) Xerox Corporation, 1991.
  * See 'implicit.eecs.wsu.edu'.
  *
  */

#include <list.h>
#include <stack.h>
#include <vector.h>
#include "vctr.h"
#include "vertex.h"
#include "../debug.h"

//using namespace :: std;

class SOLID;
class RECIEVER;

enum { poly_parametric, poly_implicit_cube, poly_implicit_tetrahedron };

const int max_iter = 10;
inline int flip_bit( int i, int bit )
   { return i ^ 1 << bit; };


enum { L = 0,  // left direction   - x
       R,      // right direction  + x
       B,      // bottom direction - y
       T,      // top direction    + y
       N,      // near direction   - z
       F       // far direction    + z
       };
enum { LB = 0, // left bottom edge
       LT,     // left top edge
       LN,     // left near edge
       LF,     // left far edge
       RB,     // right bottom edge
       RT,     // right top edge
       RN,     // right near edge
       RF,     // right far edge
       BN,     // bottom near edge
       BF,     // bottom far edge
       TN,     // top near edge
       TF      // top far edge
       };

enum { LBN = 0,   // left bottom near corner
       LBF,       // left bottom far corner
       LTN,       // left top near corner
       LTF,       // left top far corner
       RBN,       // right bottom near corner
       RBF,       // right bottom far corner
       RTN,       // right top near corner
       RTF        // right top far corner
       };

struct CORNER
{
   int i, j, k;
   VECTOR position;
   double value;
   CORNER( int _i, int _j, int _k, const VECTOR& p, const double& val )
      :i( _i ), j( _j ), k( _k ), position( p ), value( val ){};
};

struct EDGE
{
   CORNER* corners[ 2 ];
   VERTEX* vertex;
   EDGE( CORNER* c1, CORNER* c2, VERTEX* v )
      : vertex( v ){ corners[ 0 ] = c1; corners[ 1 ] = c2;};
   ~EDGE() { assert( vertex ); delete vertex; };
};

struct CUBE
{
   int i, j, k;
   // cube lattice
   CORNER* corners[ 8 ];
   CUBE( int, int, int );
};

struct CENTER
{
   int i, j, k;
   CENTER( int _i, int _j, int _k )
      : i( _i ), j( _j ), k( _k ){};
};

class POLYGONIZER
{
   public:
   POLYGONIZER();
   ~POLYGONIZER();

   //! Creates initial cubes for implicite polygonizer
   /*! We need to call this method several times in case of
       non-connected surfaces. It si at least once called automaticly
       if it was not called before starting the process of tesselation.
    */
   int Init( const VECTOR& pos, const SOLID* sld, const double& cb_sz );

   //! Method for implicite surfaces
   /*! start polygonizer with given SOLID, triangles are inserted into TRIANGLES
    two const VECTOR&s define the bound for polygonizations
    const double& is size of elementar cube
    int is poligonization mode - polygonize cube or tetrahedrons
    */
   int Implicit( const SOLID*, RECIEVER*, const VECTOR&, const VECTOR&, const double&, int );
   
   int Parametric( const SOLID*, RECIEVER*, const unsigned int, const unsigned int );
   protected:
   
   //! This solid will be polygonized using method SOLID :: Solid_Function( const VECTOR& );
   const SOLID* solid;

   //! Reciver of emitted triangles
   RECIEVER* reciever;

   //! Position of the first cube
   /*! Thus, it is also position of the origin for cubes indexing 
    */
   VECTOR position;
   
   //! Bounds for polygonizer
   /*! All coubes outside culled.
    */
   VECTOR bound_ps, bound_cr;

   //! Size of the cube and
   double cube_size;

   stack< CUBE* > cube_stack;
   vector< vector< int > > cube_table[ 256 ];
   list< EDGE* >** edge_hash;
   // hash field of edges lists
   list< CORNER* >** corner_hash;
   // another hash for corners
   list< CENTER* >** center_hash;
   // yet another hash for cube centers

   int Triangulate_Cube( CUBE* );
   // triangulate the cube directly, without decomposition
   VERTEX* Get_Vertex( CORNER*, CORNER* );
   // return vertex for given edge using edge hash table ( Get_Edge method )
   // both corners values are presumed of different sign
   VECTOR Compute_Surface_Point( const VECTOR&, const VECTOR&, const double& );
   // compute vertex on given edge using solid function of given solid
   void Set_Edge( CORNER*, CORNER*, VERTEX* );
   // insert edge to hash table
   int Check_Edge( CORNER*, CORNER*, VERTEX*& );
   // get vertex on edge from hash table
   // return 1 if edge is in hash table
   //        0 if edge is not in hash table
   CORNER* Set_Corner( int, int, int );
   // return corner with given lattice location
   // set and cache its function value
   int Set_Center( int, int, int );
   // set ( i, j, k ) entry to center_hash table
   // return 1 if already set; otherwise set and return 0
   void Make_Cube_Table();
   // create cube_table
   int Next_Clockwise_Edge( int, int );
   // return next clockwise edge from given edge around given face
   int Other_Face( int, int );
   // return face adjoining edge is not the given face
   void Test_Face( int, int, int, CUBE*, int, int, int, int, int );
   // given cube at lattice ( i, j, k ) and four corners of face,
   // if surface crosses face, compute other four corners of adjacent cube
   // and add new cube to cube stack
   int Find_Point( int, VECTOR&, double, const SOLID* sld );
   // find point with given value sign
   void Free_Memory();
};

extern POLYGONIZER polygonizer;

#endif
