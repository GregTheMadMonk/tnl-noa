/***************************************************************************
                          mMPIMesh2D.cpp  -  description
                             -------------------
    begin                : 2005/07/09
    copyright            : (C) 2005 by Tomas Oberhuber
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

#include "mMPIMesh2D.h"
#include <config.h>
#include "mdiff-cpp-headers.h"
#include "tnlGrid2D.h"
#include "boundary.h"
#include "mfuncs.h"
#include "msystem.h"
#include "debug.h"
#include "mpi-supp.h"


//--------------------------------------------------------------------------
mMPIMesh2D :: mMPIMesh2D()
   : mesh_comm( 0 ),
     mesh_x_size( 1 ),
     mesh_y_size( 1 ),
     node_x_pos( 0 ),
     node_y_pos( 0 ),
     local_x_size( 0 ),
     local_y_size( 0 ),
     global_x_size( 0 ),
     global_y_size( 0 ),
     overlap_width( 0 ),
     lower_overlap( 0 ),
     upper_overlap( 0 ),
     left_overlap( 0 ),
     right_overlap( 0 ),
     node_rank( 0 ),
     lft_snd_buf( 0 ),
     rght_snd_buf( 0 ),
     lwr_snd_buf( 0 ),
     uppr_snd_buf( 0 ),
     lwr_lft_snd_buf( 0 ),
     lwr_rght_snd_buf( 0 ),
     uppr_lft_snd_buf( 0 ),
     uppr_rght_snd_buf( 0 ),
     lft_rcv_buf( 0 ),
     rght_rcv_buf( 0 ),
     lwr_rcv_buf( 0 ),
     uppr_rcv_buf( 0 ),
     lwr_lft_rcv_buf( 0 ),
     lwr_rght_rcv_buf( 0 ),
     uppr_lft_rcv_buf( 0 ),
     uppr_rght_rcv_buf( 0 ),
     lwr_lft_nb( -1 ),
     lwr_rght_nb( -1 ),
     uppr_lft_nb( -1 ),
     uppr_rght_nb( -1 )
{
#ifdef HAVE_MPI_H
   MPI_Comm_rank( MPI_COMM_WORLD, &node_rank );
#endif
}
//--------------------------------------------------------------------------
mMPIMesh2D :: mMPIMesh2D( const mMPIMesh2D& mpi_mesh,
                          m_int _overlap_width )
   : mesh_comm( mpi_mesh. mesh_comm ),
     mesh_x_size( mpi_mesh. mesh_x_size ),
     mesh_y_size( mpi_mesh. mesh_y_size ),
     node_x_pos( mpi_mesh. node_x_pos ),
     node_y_pos( mpi_mesh. node_y_pos ),
     local_x_size( mpi_mesh. local_x_size ),
     local_y_size( mpi_mesh. local_y_size ),
     global_x_size( mpi_mesh. global_x_size ),
     global_y_size( mpi_mesh. global_y_size ),
     node_rank( mpi_mesh. node_rank ),
     lwr_lft_nb( mpi_mesh. lwr_lft_nb ),
     lwr_rght_nb( mpi_mesh. lwr_rght_nb ),
     uppr_lft_nb( mpi_mesh. uppr_lft_nb ),
     uppr_rght_nb( mpi_mesh. uppr_rght_nb )
     

{
   overlap_width = _overlap_width;
   lower_overlap = overlap_width * ( mpi_mesh. lower_overlap != 0 );
   upper_overlap = overlap_width * ( mpi_mesh. upper_overlap != 0 );
   left_overlap = overlap_width * ( mpi_mesh. left_overlap != 0 );
   right_overlap = overlap_width * ( mpi_mesh. right_overlap != 0 );
   
   m_int buf_size = overlap_width * overlap_width;
   
   if( mpi_mesh. lft_snd_buf )
      lft_snd_buf = new m_real[ local_y_size * overlap_width ];
   if( mpi_mesh. rght_snd_buf )
      rght_snd_buf = new m_real[ local_y_size * overlap_width ];
   if( mpi_mesh. lwr_snd_buf )
      lwr_snd_buf = new m_real[ local_x_size * overlap_width ];
   if( mpi_mesh. uppr_snd_buf )
      uppr_snd_buf = new m_real[ local_x_size * overlap_width ];
   if( mpi_mesh. lwr_lft_snd_buf )
      lwr_lft_snd_buf = new m_real[ buf_size ];
   if( mpi_mesh. lwr_rght_snd_buf )
      lwr_rght_snd_buf = new m_real[ buf_size ];
   if( mpi_mesh. uppr_lft_snd_buf )
      uppr_lft_snd_buf = new m_real[ buf_size ];
   if( mpi_mesh. uppr_rght_snd_buf )
      uppr_rght_snd_buf = new m_real[ buf_size ];
   if( mpi_mesh. lft_rcv_buf )
      lft_rcv_buf = new m_real[ local_y_size * overlap_width ];
   if( mpi_mesh. rght_rcv_buf )
      rght_rcv_buf = new m_real[ local_y_size * overlap_width ];
   if( mpi_mesh. lwr_rcv_buf )
      lwr_rcv_buf = new m_real[ local_x_size * overlap_width ];
   if( mpi_mesh. uppr_rcv_buf )
      uppr_rcv_buf = new m_real[ local_x_size * overlap_width ];
   if( mpi_mesh. lwr_lft_rcv_buf )
      lwr_lft_rcv_buf = new m_real[ buf_size ];
   if( mpi_mesh. lwr_rght_rcv_buf )
      lwr_rght_rcv_buf = new m_real[ buf_size ];
   if( mpi_mesh. uppr_lft_rcv_buf )
      uppr_lft_rcv_buf = new m_real[ buf_size ];
   if( mpi_mesh. uppr_rght_rcv_buf )
      uppr_rght_rcv_buf = new m_real[ buf_size ];
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: FreeBuffers()
{
   dbgFunctionName( "mMPIMesh2D", "FreeBuffers" );
   dbgExpr( lft_snd_buf );
   dbgExpr( rght_snd_buf );
   dbgExpr( uppr_snd_buf );
   dbgExpr( lwr_snd_buf );
   dbgExpr( uppr_lft_snd_buf );
   dbgExpr( uppr_rght_snd_buf );
   dbgExpr( lwr_lft_snd_buf );
   dbgExpr( lwr_rght_snd_buf );
   dbgExpr( lft_rcv_buf );
   dbgExpr( rght_rcv_buf );
   dbgExpr( uppr_rcv_buf );
   dbgExpr( lwr_rcv_buf );
   dbgExpr( uppr_lft_rcv_buf );
   dbgExpr( uppr_rght_rcv_buf );
   dbgExpr( lwr_lft_rcv_buf );
   dbgExpr( lwr_rght_rcv_buf );

   if( lft_snd_buf ) delete[] lft_snd_buf;
   if( rght_snd_buf ) delete[] rght_snd_buf;
   if( lwr_snd_buf ) delete[] lwr_snd_buf;
   if( uppr_snd_buf ) delete[] uppr_snd_buf;
   if( lwr_lft_snd_buf ) delete[] lwr_lft_snd_buf;
   if( lwr_rght_snd_buf ) delete[] lwr_rght_snd_buf;
   if( uppr_lft_snd_buf ) delete[] uppr_lft_snd_buf;
   if( uppr_rght_snd_buf ) delete[] uppr_rght_snd_buf;
   if( lft_rcv_buf ) delete[] lft_rcv_buf;
   if( rght_rcv_buf ) delete[] rght_rcv_buf;
   if( lwr_rcv_buf ) delete[] lwr_rcv_buf;
   if( uppr_rcv_buf ) delete[] uppr_rcv_buf;
   if( lwr_lft_rcv_buf ) delete[] lwr_lft_rcv_buf;
   if( lwr_rght_rcv_buf ) delete[] lwr_rght_rcv_buf;
   if( uppr_lft_rcv_buf ) delete[] uppr_lft_rcv_buf;
   if( uppr_rght_rcv_buf ) delete[] uppr_rght_rcv_buf;
   lft_snd_buf = rght_snd_buf = lwr_snd_buf = uppr_snd_buf =
   lwr_lft_snd_buf = lwr_rght_snd_buf = 
   uppr_lft_snd_buf = uppr_rght_snd_buf =
   lft_rcv_buf = rght_rcv_buf = lwr_rcv_buf = uppr_rcv_buf =
   lwr_lft_rcv_buf = lwr_rght_rcv_buf = 
   uppr_lft_rcv_buf = uppr_rght_rcv_buf = 0;

}
//--------------------------------------------------------------------------
mMPIMesh2D :: ~mMPIMesh2D()
{
   FreeBuffers();
}
//--------------------------------------------------------------------------
m_int mMPIMesh2D :: Init( const tnlGrid2D* phi,
                          m_int _mesh_x_size,
                          m_int _mesh_y_size,
                          m_int _overlap_width )
{
   dbgFunctionName( "mMPIMesh2D", "Init" );
   
   // for consistency we want to have global dimensions set even
   // if we do not use mpi at all
   dbgExpr( node_rank );
   if( node_rank == 0 )
   {
      assert( phi );
      global_x_size = phi -> XSize();
      global_y_size = phi -> YSize();
   }
   
   // if nothing happens we want mesh size to be 1
   mesh_x_size = mesh_y_size = 1;
#ifdef HAVE_MPI_H
   
   m_int nproc;
   MPI_Comm_size( MPI_COMM_WORLD, &nproc );
   if( _mesh_x_size == 1 && _mesh_y_size == 1 ) return true;
   if( nproc == 0 && _mesh_x_size == 0 && _mesh_y_size == 0 ) return true;


   dbgExpr( node_rank );
   m_int dims[ 2 ] = { 0, 0 };
   m_int periods[ 2 ] = { 0, 0 };
   if( ! _mesh_x_size && ! _mesh_y_size )
   {
      dbgCout( "Finding mesh dimensions..." );
      m_int nproc;
      MPI_Comm_size( MPI_COMM_WORLD, &nproc );
      MPI_Dims_create( nproc, 2, dims );
      mesh_x_size = dims[ 0 ];
      mesh_y_size = dims[ 1 ];
   }
   else
   {
      dims[ 0 ] = mesh_x_size = _mesh_x_size;
      dims[ 1 ] = mesh_y_size = _mesh_y_size;
   }
   dbgExpr( mesh_x_size );
   dbgExpr( mesh_y_size );
       

   MPI_Cart_create( MPI_COMM_WORLD, 
                    2,  // ndims
                    dims,
                    periods,
                    true,
                    &mesh_comm );
   if( mesh_comm == MPI_COMM_NULL )
   {
      cerr << "Not enough nodes for creating mesh " << mesh_x_size <<
              "x" << mesh_y_size << endl;
      return 0;
   }
   DBG_MPI_BARRIER;
   dbgExpr( mesh_comm );

   overlap_width = _overlap_width;
   if( node_rank == 0 )
   {
      if( ! mesh_x_size || ! mesh_y_size )
      {
         cerr << "One of MPI mesh dimension is 0. " << endl;
         return 0;
      }
      // check for grid size and parallel splitting
      assert( phi );
      global_x_size = phi -> XSize();
      global_y_size = phi -> YSize();
      if( global_x_size % mesh_x_size != 0 || 
          global_y_size % mesh_y_size != 0 )
      {
         cerr << "Wrong numbers of subdomains for splitting. They do not divide the dimensions of the grid." << endl;
         return 0;
      }
      local_x_size = global_x_size / mesh_x_size;
      local_y_size = global_y_size / mesh_y_size;
   }
   MPIBcast< m_int >( &local_x_size, 0 );
   MPIBcast< m_int >( &local_y_size, 0 );
   MPIBcast< m_int >( &global_x_size, 0 );
   MPIBcast< m_int >( &global_y_size, 0 );
   MPIBcast< m_int >( &overlap_width, 0 );
  
   m_int coords[ 2 ];
   MPI_Comm_rank( mesh_comm, &node_rank );
   MPI_Cart_coords( mesh_comm, node_rank, 2, coords ); 
   node_x_pos = coords[ 0 ];
   node_y_pos = coords[ 1 ];

   m_int nb_node;
   left_overlap = right_overlap = upper_overlap = lower_overlap = 0;
   
   dbgCout( "Freeing buffers..." );
   FreeBuffers();
   DBG_MPI_BARRIER;
   dbgCout( "Freeing buffers...done." );
   
   if( LeftNeighbour( nb_node ) )
   {
      left_overlap = overlap_width;
      lft_snd_buf = new m_real[ local_y_size * overlap_width ];
      lft_rcv_buf = new m_real[ local_y_size * overlap_width ];
   }
   if( RightNeighbour( nb_node ) )
   {
      right_overlap = overlap_width;
      rght_snd_buf = new m_real[ local_y_size * overlap_width ];
      rght_rcv_buf = new m_real[ local_y_size * overlap_width ];
   }
   if( LowerNeighbour( nb_node ) )
   {
      lower_overlap = overlap_width;
      lwr_snd_buf = new m_real[ local_x_size * overlap_width ];
      lwr_rcv_buf = new m_real[ local_x_size * overlap_width ];
   }
   if( UpperNeighbour( nb_node ) )
   {
      upper_overlap = overlap_width;
      uppr_snd_buf = new m_real[ local_x_size * overlap_width ];
      uppr_rcv_buf = new m_real[ local_x_size * overlap_width ];
   }

   m_int buf_size = overlap_width * overlap_width;
   if( lower_overlap && left_overlap )
   {
      lwr_lft_snd_buf = new m_real[ buf_size ];
      lwr_lft_rcv_buf = new m_real[ buf_size ];
      lwr_lft_nb = FindNodeByCoord( mesh_comm, node_x_pos - 1, node_y_pos - 1 );
   }
   if( lower_overlap && right_overlap )
   {
      lwr_rght_snd_buf = new m_real[ buf_size ];
      lwr_rght_rcv_buf = new m_real[ buf_size ];
      lwr_rght_nb = FindNodeByCoord( mesh_comm, node_x_pos + 1, node_y_pos - 1 );
   }
   if( upper_overlap && left_overlap )
   {
      uppr_lft_snd_buf = new m_real[ buf_size ];
      uppr_lft_rcv_buf = new m_real[ buf_size ];
      uppr_lft_nb = FindNodeByCoord( mesh_comm, node_x_pos - 1, node_y_pos + 1 );
   }
   if( upper_overlap && right_overlap )
   {
      uppr_rght_snd_buf = new m_real[ buf_size ];
      uppr_rght_rcv_buf = new m_real[ buf_size ];
      uppr_rght_nb = FindNodeByCoord( mesh_comm, node_x_pos + 1, node_y_pos + 1 );
   }

   dbgCout( " node rank " << node_rank << 
             " ( " << node_x_pos << ", " << node_y_pos << " ), LRLU nb " <<
             left_overlap << right_overlap <<
             lower_overlap << upper_overlap );
   dbgExpr( lwr_lft_nb );
   dbgExpr( uppr_rght_nb );
   dbgExpr( uppr_lft_nb );
   dbgExpr( lwr_rght_nb );
   


#else
   cerr << "WARNING: Missing support for OpenMPI. If you really need it you should recompile mDiff library." << endl;
#endif
   return 1;
}
//--------------------------------------------------------------------------
m_bool mMPIMesh2D :: CreateMesh( const tnlGrid2D* phi,
                                 tnlGrid2D*& sub_phi ) const
{
   dbgFunctionName( "mMPIMesh2D", "CreateMesh" );
#ifdef HAVE_MPI_H
   if( MeshSize() == 1 ) return true;
   tnlVector2D a, h;
   tnlString name;
   if( node_rank == 0 )
   {
      assert( phi );
      h = phi -> H();
      a = phi -> A();
      name = phi -> GetName();
   }
   MPIBcast< tnlVector2D >( &h, 0, 1, mesh_comm );
   MPIBcast< tnlVector2D >( &a, 0, 1, mesh_comm );
   name. MPIBcast( 0, mesh_comm );
   sub_phi = new tnlGrid2D( a,
                          tnlVector2D( 0.0, 0.0 ), // 'b' is not important now
                          h,
                          local_x_size + left_overlap + right_overlap, // x-size
                          local_y_size + lower_overlap + upper_overlap, // y-size
                          node_x_pos * local_x_size - left_overlap, // x-offset
                          node_y_pos * local_y_size - lower_overlap, // y-offset
                          name. Data()
         );

   dbgCout( "( Rank " << node_rank << " ) - creating subdomain < " <<
               node_x_pos * local_x_size - left_overlap << ", " <<
               ( node_x_pos + 1 ) * local_x_size + right_overlap << " > x < " <<
               node_y_pos * local_y_size - lower_overlap << ", " <<
               ( node_y_pos + 1 ) * local_y_size + upper_overlap << " >" );

   if( ! sub_phi )
   {
      cerr << "Not enough memory to allocate new subdomain on node " << node_rank << endl;
      return 0;
   }
#endif
   return 1;
}
//--------------------------------------------------------------------------
m_bool mMPIMesh2D :: CreateMeshAtBoundaries( const tnlGrid2D* phi,
                                             tnlGrid2D*& sub_phi )
{
#ifdef HAVE_MPI_H
   if( MeshSize() == 1 ) return true;
   // if there is no source function do nothing
   m_int quit( 0 );
   if( node_rank == 0 && ! phi )
      quit = 1;
   MPI_Bcast( &quit, 1, MPI_INT, 0, mesh_comm );
   if( quit )
   {
      sub_phi = 0;
      return true;
   }
   if( ! CreateMesh( phi, sub_phi ) )
      return false;
   MPI_Barrier( mesh_comm );
   if( node_x_pos != 0 &&
       node_x_pos != mesh_x_size - 1 &&
       node_y_pos != 0 &&
       node_y_pos != mesh_y_size - 1 )
   {
      delete sub_phi;
      sub_phi = 0;
   }
#endif
   return true;
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: CreateGlobalGrid( tnlGrid2D*& phi,
                                     const tnlGrid2D* sub_phi ) const
{
   dbgFunctionName( "mMPIMesh2D", "CreateGlobalGrid" );
   dbgExpr( node_rank );
   dbgExpr( global_x_size );
   dbgExpr( global_y_size );
   phi = new tnlGrid2D( sub_phi -> A(),
                      tnlVector2D( 0.0, 0.0 ),
                      sub_phi -> H(),
                      global_x_size,
                      global_y_size,
                      0,
                      0,
                      sub_phi -> GetName(). Data () );
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: ScatterToNode( const tnlGrid2D* phi,
                                  tnlGrid2D* sub_phi,
                                  m_int dest_node ) const
{
   dbgFunctionName( "mMPIMesh2D", "ScatterToNode" );
#ifdef HAVE_MPI_H
   if( node_rank == 0 )
   {
      m_int dest_x_pos;
      m_int dest_y_pos;
      m_int coords[ 2 ];
      MPI_Cart_coords( mesh_comm, dest_node, 2, coords );
      dest_x_pos = coords[ 0 ];
      dest_y_pos = coords[ 1 ];

      m_int dest_left_overlap( 0 ), dest_right_overlap( 0 ),
            dest_lower_overlap( 0 ), dest_upper_overlap( 0 );
      m_int nb_node;
      dbgExpr( dest_node );
      if( dest_x_pos > 0 ) dest_left_overlap = overlap_width;
      if( dest_x_pos < mesh_x_size - 1 ) dest_right_overlap = overlap_width;
      if( dest_y_pos > 0 ) dest_lower_overlap = overlap_width;
      if( dest_y_pos < mesh_y_size - 1 ) dest_upper_overlap = overlap_width;
     
      dbgCout( "dest edges LRLU " << 
               dest_left_overlap << dest_right_overlap <<
               dest_lower_overlap << dest_upper_overlap );
      tnlGrid2D mpi_buff( tnlVector2D( 0.0, 0.0 ), // a
                        tnlVector2D( 0.0, 0.0 ), // b
                        tnlVector2D( 1.0, 1.0 ), // h
                        local_x_size + dest_left_overlap + dest_right_overlap, // x-size
                        local_y_size + dest_lower_overlap + dest_upper_overlap, // y-size
                        dest_x_pos * local_x_size - dest_left_overlap, // x-offset
                        dest_y_pos * local_y_size - dest_lower_overlap // y-offset
                        );
      dbgCout( "mpi buff < " <<
                  dest_x_pos * local_x_size - dest_left_overlap << ", " <<
                  ( dest_x_pos + 1 ) * local_x_size + dest_right_overlap << " > x < " <<
                  dest_y_pos * local_y_size - dest_lower_overlap << ", " <<
                  ( dest_y_pos + 1 ) * local_y_size + dest_upper_overlap << " >" );
      
      m_int i, j;
      const m_int i1 = dest_x_pos * local_x_size - dest_left_overlap;
      const m_int i2 = ( dest_x_pos + 1 ) * local_x_size + dest_right_overlap;
      const m_int j1 = dest_y_pos * local_y_size - dest_lower_overlap;
      const m_int j2 = ( dest_y_pos + 1 ) * local_y_size + dest_upper_overlap;
      for( i = i1; i < i2; i ++ )
         for( j = j1; j < j2; j ++ )
            mpi_buff( i, j ) = ( * phi )( i, j );
      m_int buf_size = 
         ( local_x_size + dest_left_overlap + dest_right_overlap ) *
         ( local_y_size + dest_lower_overlap + dest_upper_overlap );
      MPI_Send( mpi_buff. Data(),
                buf_size * sizeof( m_real ),
                MPI_CHAR,
                dest_node,
                0,
                mesh_comm );
   }
   else
   {
      m_int buf_size =
            ( local_x_size + left_overlap + right_overlap ) *
            ( local_y_size + lower_overlap + upper_overlap );
      MPI_Status status;
      MPI_Recv( sub_phi -> Data(),
                buf_size * sizeof( m_real ),
                MPI_CHAR,
                0,
                0,
                mesh_comm,
                &status );
      //cout << "Receiving data on node " << node_rank << endl;
   }
#endif
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: Scatter( const tnlGrid2D* phi, tnlGrid2D* sub_phi ) const
{
   if( MeshSize() == 1 ) return;
   if( node_rank == 0 )
   {
      m_int dest, mesh_size = MeshSize();
      m_int i, j;
      for( dest = 1; dest < mesh_size; dest ++ )
      {
         ScatterToNode( phi, sub_phi, dest );
         //cout << "Sending data to node " << dest << endl;
      }
      for( i = 0; i < local_x_size + right_overlap; i ++ )
         for( j = 0; j < local_y_size + upper_overlap; j ++ )
            ( * sub_phi )( i, j ) = ( * phi )( i, j );
   }
   else ScatterToNode( phi, sub_phi, 0 );
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: ScatterAtBoundaries( const tnlGrid2D* phi,
                                        tnlGrid2D* sub_phi )
{
   dbgFunctionName( "mMPIMesh2D", "ScatterAtBoundaries" );
   if( MeshSize() == 1 ) return;
   if( ! sub_phi ) return;
#ifdef HAVE_MPI_H
   dbgExpr( node_rank );
   if( node_rank == 0 )
   {
      m_int dest, coords[ 2 ];
      for( dest = 0; dest < MeshSize(); dest ++ )
      {
         MPI_Cart_coords( mesh_comm, dest, 2, coords );
         if( coords[ 0 ] == 0 || coords[ 0 ] == mesh_x_size - 1 ||
             coords[ 1 ] == 0 || coords[ 1 ] == mesh_y_size - 1 )
            ScatterToNode( phi, sub_phi, dest );
      }
      m_int i, j;
      for( i = 0; i < local_x_size + right_overlap; i ++ )
         for( j = 0; j < local_y_size + upper_overlap; j ++ )
            ( * sub_phi )( i, j ) = ( * phi )( i, j );
   }
   else
   {
      assert( node_x_pos == 0 ||
              node_x_pos == mesh_x_size - 1 ||
              node_y_pos == 0 ||
              node_y_pos == mesh_y_size - 1 );
      ScatterToNode( phi, sub_phi, 0 );
   }
#endif
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: Gather( tnlGrid2D* phi,
                           const tnlGrid2D* sub_phi ) const
{
   dbgFunctionName( "mMPIMesh2D", "Gather" );
#ifdef HAVE_MPI_H
   if( MeshSize() == 1 ) return;
   
   DBG_MPI_BARRIER;
   dbgExpr( node_rank );
   if( node_rank == 0 )
   {
      m_int nproc, src;
      m_int i, j;
      MPI_Status status;
      for( src = 1; src < MeshSize(); src ++ )
      {
         m_int coords[ 2 ];
         MPI_Cart_coords( mesh_comm, src, 2, coords ); 
         m_int src_x_pos = coords[ 0 ];
         m_int src_y_pos = coords[ 1 ];

         dbgExpr( src );
         dbgExpr( src_x_pos );
         dbgExpr( src_y_pos );

         m_int src_left_overlap, src_right_overlap,
             src_lower_overlap, src_upper_overlap;
         if( src_x_pos == 0 ) src_left_overlap = 0;
         else src_left_overlap = overlap_width;
         if( src_x_pos == mesh_x_size - 1 ) src_right_overlap = 0;
         else src_right_overlap = overlap_width;
         if( src_y_pos == 0 ) src_lower_overlap = 0;
         else src_lower_overlap = overlap_width;
         if( src_y_pos == mesh_y_size - 1 ) src_upper_overlap = 0;
         else src_upper_overlap = overlap_width;
         dbgCout( "Allocating supporting buffer < " <<
                   src_x_pos * local_x_size - src_left_overlap <<
                   ", " << ( src_x_pos + 1 ) * local_x_size + src_right_overlap <<
                   " >x< " << src_y_pos * local_y_size - src_lower_overlap <<
                   ", " << ( src_y_pos + 1 ) * local_y_size + src_upper_overlap <<
                   " >" );
               
         tnlGrid2D mpi_buff( tnlVector2D( 0.0, 0.0 ), // a
                           tnlVector2D( 0.0, 0.0 ), // b
                           tnlVector2D( 1.0, 1.0 ), // h
                           local_x_size + src_left_overlap + src_right_overlap, // x-size
                           local_y_size + src_lower_overlap + src_upper_overlap, // y-size
                           src_x_pos * local_x_size - src_left_overlap, // x-offset
                           src_y_pos * local_y_size - src_lower_overlap  // y-offset
                           );
         m_int buf_size = 
            ( local_x_size + src_left_overlap + src_right_overlap ) *
            ( local_y_size + src_lower_overlap + src_upper_overlap );
         dbgExpr( buf_size );
         dbgCout( "RECEIVING data from node " << src  );
         MPI_Recv( mpi_buff. Data(),
                   buf_size * sizeof( m_real ),
                   MPI_BYTE,
                   src,
                   0,
                   mesh_comm,
                   &status );
         dbgCout( "Receiving data done." );
         for( i = src_x_pos * local_x_size; 
              i < ( src_x_pos + 1 ) * local_x_size;
              i ++ )
            for( j = src_y_pos * local_y_size;
                 j < ( src_y_pos + 1 ) * local_y_size;
                 j ++ )
                ( * phi )( i, j ) = mpi_buff( i, j );
      }
      for( i = 0; i < local_x_size; i ++ )
         for( j = 0; j < local_y_size; j ++ )
            ( * phi )( i, j ) = ( * sub_phi )( i, j );
   }
   else
   {
      dbgCout( "node rank " << node_rank << " SENDING data" );
      m_int buf_size = ( local_x_size + left_overlap + right_overlap ) *
                       ( local_y_size + lower_overlap + upper_overlap );
      MPI_Send( const_cast< m_real* >( sub_phi -> Data() ),
                buf_size * sizeof( m_real ),
                MPI_BYTE,
                0,
                0,
                mesh_comm );
      dbgCout( "sending data done." );
   }
   dbgCout( "Gathering data done." );
#endif
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: Synchronize( tnlGrid2D* phi )
{
   dbgFunctionName( "mMPIMesh2D", "Synchronize" );
   //return;
#ifdef HAVE_MPI_H
   if( MeshSize() == 1 ) return;
  m_int i, j;
  m_int min_x = node_x_pos * local_x_size;
  m_int min_y = node_y_pos * local_y_size;
  m_int max_x = min_x + local_x_size;
  m_int max_y = min_y + local_y_size;
  m_int wdth = overlap_width;
  MPI_Status status;
   
  MPI_Request lft_snd_rqst, rght_snd_rqst, lwr_snd_rqst, uppr_snd_rqst,
              lwr_lft_snd_rqst, lwr_rght_snd_rqst,
              uppr_lft_snd_rqst, uppr_rght_snd_rqst,
              lft_rcv_rqst, rght_rcv_rqst, lwr_rcv_rqst, uppr_rcv_rqst,
              lwr_lft_rcv_rqst, lwr_rght_rcv_rqst,
              uppr_lft_rcv_rqst, uppr_rght_rcv_rqst;
  
  
  DBG_MPI_BARRIER;
  
  m_int dest_rank, src_rank; 
  //it seems thatMPI_Cart_shift can change src_rank
  
  //starting communication with the left neighbour
  if( node_x_pos > 0 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING data to the LEFT neighbour" );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < local_y_size; j ++ )
           lft_snd_buf[ i * local_y_size + j ] =
              ( * phi )( i + min_x, j + min_y );
     src_rank = node_rank;
     MPI_Cart_shift( mesh_comm, 0, -1, &src_rank, &dest_rank );
     MPI_Isend( lft_snd_buf,
                wdth * local_y_size * sizeof( m_real ),
                MPI_CHAR,
                dest_rank,
                0,
                mesh_comm ,
                &lft_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING data from the LEFT neighbour" );
     MPI_Irecv( lft_rcv_buf,
                wdth * local_y_size * sizeof( m_real ),
                MPI_CHAR,
                dest_rank,
                0,
                mesh_comm,
                &lft_rcv_rqst );
  }
  DBG_MPI_BARRIER;
  
  // starting communication with the right neighbour
  if( node_x_pos < mesh_x_size - 1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING data to the RIGHT neighbour" );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < local_y_size; j ++ )
           rght_snd_buf[ i * local_y_size + j ] =
              ( * phi )( max_x - wdth + i, j + min_y );
     src_rank = node_rank;
     MPI_Cart_shift( mesh_comm, 0, 1, &src_rank, &dest_rank );
     MPI_Isend( rght_snd_buf,
               wdth * local_y_size * sizeof( m_real ),
               MPI_CHAR,
               dest_rank,
               0,
               mesh_comm,
               &rght_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING data from the RIGHT neighbour" );
     MPI_Irecv( rght_rcv_buf,
                wdth * local_y_size * sizeof( m_real ),
                MPI_CHAR,
                dest_rank,
                0,
                mesh_comm,
                &rght_rcv_rqst );
  }
  DBG_MPI_BARRIER;
  
  // starting communication with the lower neighbour
  if( node_y_pos > 0 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING data to the LOWER neighbour" );
     for( i = 0; i < local_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           lwr_snd_buf[ j * local_x_size + i ] =
              ( * phi )( min_x + i, min_y + j);
     src_rank = node_rank;
     MPI_Cart_shift( mesh_comm, 1, -1, &src_rank, &dest_rank );
     MPI_Isend( lwr_snd_buf,
               wdth * local_x_size * sizeof( m_real ),
               MPI_CHAR,
               dest_rank,
               0,
               mesh_comm,
               &lwr_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING data from the LOWER neighbour" );
     MPI_Irecv( lwr_rcv_buf,
               wdth * local_y_size * sizeof( m_real ),
               MPI_CHAR,
               dest_rank,
               0,
               mesh_comm,
               &lwr_rcv_rqst );
  }
  DBG_MPI_BARRIER;

  // starting communication with the uppper neighbour
  if( node_y_pos < mesh_y_size - 1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING data to the UPPER neighbour" );
     for( i = 0; i < local_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           uppr_snd_buf[ j * local_x_size + i ] =
              ( * phi )( min_x + i, max_y - wdth + j);
     src_rank = node_rank;
     MPI_Cart_shift( mesh_comm, 1, 1, &src_rank, &dest_rank );
     MPI_Isend( uppr_snd_buf,
                wdth * local_x_size * sizeof( m_real ),
                MPI_CHAR,
                dest_rank,
                0,
                mesh_comm,
                &uppr_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING data from the UPPER neighbour" );
     MPI_Irecv( uppr_rcv_buf,
                wdth * local_y_size * sizeof( m_real ),
                MPI_CHAR,
                dest_rank,
                0,
                mesh_comm,
                &uppr_rcv_rqst );
  }
  DBG_MPI_BARRIER;
  
  m_int wdth_2 = wdth * wdth;
  m_int coords[ 2 ];
  
  // starting communication with lower left neighbour
  if( lwr_lft_nb != -1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the LOWER LEFT neighbour." );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           lwr_lft_snd_buf[ j * wdth + i ] =
              ( * phi )( min_x + i, min_y + j );
     MPI_Isend( lwr_lft_snd_buf,
                wdth_2 * sizeof( m_real ),
                MPI_CHAR,
                lwr_lft_nb,
                0,
                mesh_comm,
                &lwr_lft_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING small square from the LOWER LEFT neighbour." );
     MPI_Irecv( lwr_lft_rcv_buf,
                wdth_2 * sizeof( m_real ),
                MPI_CHAR,
                lwr_lft_nb,
                0,
                mesh_comm,
                &lwr_lft_rcv_rqst );
  }
  DBG_MPI_BARRIER;
  
  // starting communication with lower right neighbour
  if( lwr_rght_nb != -1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the LOWER RIGHT neighbour." );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           lwr_rght_snd_buf[ j * wdth + i ] =
              ( * phi )( max_x - wdth + i, min_y + j );
     MPI_Isend( lwr_rght_snd_buf,
                wdth_2 * sizeof( m_real ),
                MPI_CHAR,
                lwr_rght_nb,
                0,
                mesh_comm,
                &lwr_rght_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING small square from the LOWER RIGHT neighbour." );
     MPI_Irecv( lwr_rght_rcv_buf,
                wdth_2 * sizeof( m_real ),
                MPI_CHAR,
                lwr_rght_nb,
                0,
                mesh_comm,
                &lwr_rght_rcv_rqst );
  }
  DBG_MPI_BARRIER;

  // starting communication with upper left neighbour
  if( uppr_lft_nb != - 1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the UPPER LEFT neighbour." );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           uppr_lft_snd_buf[ j * wdth + i ] =
              ( * phi )( min_x + i, max_y - wdth + j );
     MPI_Isend( uppr_lft_snd_buf,
                wdth_2 * sizeof( m_real ),
                MPI_CHAR,
                uppr_lft_nb,
                0,
                mesh_comm,
                &uppr_lft_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING small square from the UPPER LEFT neighbour." );
     MPI_Irecv( uppr_lft_rcv_buf,
                wdth_2 * sizeof( m_real ),
                MPI_CHAR,
                uppr_lft_nb,
                0,
                mesh_comm,
                &uppr_lft_rcv_rqst );
  }
  DBG_MPI_BARRIER;

  // starting communication with upper right neighbour
  if( uppr_rght_nb != -1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the UPPER RIGHT neighbour." );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           uppr_rght_snd_buf[ j * wdth + i ] =
              ( * phi )( max_x - wdth + i, max_y - wdth + j );
     MPI_Isend( uppr_rght_snd_buf,
                wdth_2 * sizeof( m_real ),
                MPI_CHAR,
                uppr_rght_nb,
                0,
                mesh_comm,
                &uppr_rght_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING small square from the UPPER RIGHT neighbour." );
     MPI_Irecv( uppr_rght_rcv_buf,
                wdth_2 * sizeof( m_real ),
                MPI_CHAR,
                uppr_rght_nb,
                0,
                mesh_comm,
                &uppr_rght_rcv_rqst );
  }
  DBG_MPI_BARRIER;

  // finishing communication with the left neighbour
  if( node_x_pos > 0 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from LEFT neighbour." );
     MPI_Wait( &lft_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < local_y_size; j ++ )
           ( * phi )( min_x - wdth + i, min_y + j ) =
           lft_rcv_buf[ i * local_y_size + j ];
     MPI_Wait( &lft_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;
  
  // finishing communication with the right neighbour
  if( node_x_pos < mesh_x_size - 1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from RIGHT neighbour." );
     MPI_Wait( &rght_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < local_y_size; j ++ )
           ( * phi )( max_x + i, min_y + j ) =
           rght_rcv_buf[ i * local_y_size + j ];
     MPI_Wait( &rght_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;
  
  // finishing communication with the lower neighbour
  if( node_y_pos > 0 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from LOWER neighbour." );
     MPI_Wait( &lwr_rcv_rqst, &status );
     for( i = 0; i < local_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           ( * phi )( min_x + i, min_y - wdth + j ) =
           lwr_rcv_buf[ j * local_x_size + i ];
     MPI_Wait( &lwr_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;
  
  // finishing communication with the upper neighbour
  if( node_y_pos < mesh_y_size - 1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from UPPER neighbour." );
     MPI_Wait( &uppr_rcv_rqst, &status );
     for( i = 0; i < local_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           ( * phi )( min_x + i, max_y + j ) =
           uppr_rcv_buf[ j * local_x_size + i ];
     MPI_Wait( &uppr_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;
  
  
  // finishing communication with the lower left neighbour
  if( lwr_lft_nb != -1  )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from LOWER LEFT neighbour." );
     MPI_Wait( &lwr_lft_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           ( * phi )( min_x - wdth + i, min_y - wdth + j ) =
           lwr_lft_rcv_buf[ j * wdth + i ];
     MPI_Wait( &lwr_lft_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;
 
  // finishing communication with the lower right neighbour
  if( lwr_rght_nb != -1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from LOWER RIGHT neighbour." );
     MPI_Wait( &lwr_rght_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           ( * phi )( max_x + i, min_y - wdth + j ) =
           lwr_rght_rcv_buf[ j * wdth + i ];
     MPI_Wait( &lwr_rght_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;

  // finishing communication with the upper right neighbour
  if( uppr_rght_nb != -1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from UPPER RIGHT neighbour." );
     MPI_Wait( &uppr_rght_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           ( * phi )( max_x + i, max_y + j ) =
           uppr_rght_rcv_buf[ j * wdth + i ];
     MPI_Wait( &uppr_rght_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;
  
  // finishing communication with the upper left neighbour
  if( uppr_lft_nb != -1 )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from UPPER LEFT neighbour." );
     MPI_Wait( &uppr_lft_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           ( * phi )( min_x - wdth + i, max_y + j ) =
           uppr_lft_rcv_buf[ j * wdth + i ];
     MPI_Wait( &uppr_lft_snd_rqst, &status );
  }
  
  dbgCout( "Synchronisation done..." );
  DBG_MPI_BARRIER;
#endif
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: SetDirichletBnd( tnlGrid2D* phi, const tnlGrid2D* bnd )
{
   if( MeshSize() == 1 )
   {
      :: SetDirichletBnd( phi, bnd );
      return;
   }
   m_int i, j;
   m_int i1 = node_x_pos * local_x_size - left_overlap;
   m_int i2 = ( node_x_pos + 1 ) * local_x_size + right_overlap;
   m_int j1 = node_y_pos * local_y_size - lower_overlap;
   m_int j2 = ( node_y_pos + 1 ) * local_y_size + upper_overlap;
   if( node_y_pos == 0 )
      for( i = i1; i < i2; i ++ )
         ( *phi )( i, 0 ) = ( *bnd )( i, 0 );
   if( node_y_pos == mesh_y_size - 1 )
      for( i = i1; i < i2; i ++ )
         ( *phi )( i, global_y_size - 1 ) = ( *bnd )( i, global_y_size - 1 );
   if( node_x_pos == 0 )
      for( j = j1; j < j2; j ++ )
         ( *phi )( 0, j ) = ( *bnd )( 0, j );
   if( node_x_pos == mesh_x_size - 1 )
      for( j = j1; j < j2; j ++ )
         ( *phi )( global_x_size - 1, j ) = ( *bnd )( global_x_size - 1, j );
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: SetNeumannBnd( tnlGrid2D* phi, const tnlGrid2D* bnd )
{
   if( MeshSize() == 1 )
   {
      :: SetNeumannBnd( phi, bnd );
      return;
   }
   m_int i, j;
   m_int i1 = node_x_pos * local_x_size - left_overlap;
   m_int i2 = ( node_x_pos + 1 ) * local_x_size + right_overlap;
   m_int j1 = node_y_pos * local_y_size - lower_overlap;
   m_int j2 = ( node_y_pos + 1 ) * local_y_size + upper_overlap;
   const m_real hx = phi -> H(). x;
   const m_real hy = phi -> H(). y;
   if( node_y_pos == 0 )
      for( i = i1; i < i2; i ++ )
         ( *phi )( i, 0 ) = ( *phi )( i, 1 ) - hy * ( *bnd )( i, 0 );
   if( node_y_pos == mesh_y_size - 1 )
      for( i = i1; i < i2; i ++ )
         ( *phi )( i, global_y_size - 1 ) = 
            ( *phi )( i, global_y_size - 2 ) +
            hy * ( *bnd )( i, global_y_size - 1 );
   if( node_x_pos == 0 )
      for( j = j1; j < j2; j ++ )
         ( *phi )( 0, j ) = ( *phi )( 1, j ) - hx * ( *bnd )( 0, j );
   if( node_x_pos == mesh_x_size - 1 )
      for( j = j1; j < j2; j ++ )
         ( *phi )( global_x_size - 1, j ) = 
            ( *phi )( global_x_size - 2, j ) + 
            hx * ( *bnd )( global_x_size - 1, j );
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: DomainDimensions( m_int& x_pos, m_int& y_pos,
                                     m_int& x_size, m_int& y_size )
{
   x_pos = node_x_pos * local_x_size;
   y_pos = node_y_pos * local_y_size;
   x_size = local_x_size;
   y_size = local_y_size;
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: GlobalDimensions( m_int& x_size, m_int& y_size )
{
   x_size = global_x_size;
   y_size = global_y_size;
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: DomainEdges( m_int& right, m_int& left,
                                m_int& bottom, m_int& top )
{
   right = right_overlap;
   left = left_overlap;
   bottom = lower_overlap;
   top = upper_overlap;
}
//--------------------------------------------------------------------------
m_int mMPIMesh2D :: NodeRank() const
{ 
   return node_rank;
}
//--------------------------------------------------------------------------
m_int mMPIMesh2D :: XPos() const
{
   return node_x_pos;
}
//--------------------------------------------------------------------------
m_int mMPIMesh2D :: YPos() const
{
   return node_y_pos;
}
//--------------------------------------------------------------------------
m_int mMPIMesh2D :: MeshSize() const
{
   return mesh_x_size * mesh_y_size;
}
//--------------------------------------------------------------------------
m_int mMPIMesh2D :: XSize() const
{
   return mesh_x_size;
}
//--------------------------------------------------------------------------
m_int mMPIMesh2D :: YSize() const
{
   return mesh_y_size;
}
//--------------------------------------------------------------------------
m_bool mMPIMesh2D :: LeftNeighbour( m_int& nb_node ) const
{
   return :: LeftNeighbour( mesh_comm, node_rank, nb_node );
}
//--------------------------------------------------------------------------
m_bool mMPIMesh2D :: RightNeighbour( m_int& nb_node ) const
{
   return :: RightNeighbour( mesh_comm, node_rank, nb_node );
}
//--------------------------------------------------------------------------
m_bool mMPIMesh2D :: UpperNeighbour( m_int& nb_node ) const
{
   return :: UpperNeighbour( mesh_comm, node_rank, nb_node );
}
//--------------------------------------------------------------------------
m_bool mMPIMesh2D :: LowerNeighbour( m_int& nb_node ) const
{
   return :: LowerNeighbour( mesh_comm, node_rank, nb_node );
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: DrawFunction( const tnlGrid2D* phi, const m_char* file_name )
{
   if( MeshSize() == 1 )
   {
      assert( phi );
      phi -> DrawFunction( file_name );
      return;
   }
   tnlGrid2D* f;
   if( NodeRank() == 0 )
   {
      tnlVector2D a = phi -> A();
      tnlVector2D h = phi -> H();
      f = new tnlGrid2D( a, // a
                       tnlVector2D( 0.0, 0.0 ), //b
                       h, // h 
                       local_x_size * mesh_x_size, // x-size
                       local_y_size * mesh_y_size // y-size
                      );
   }
   Gather( f, phi );
   if( NodeRank() == 0 )
   {
      f -> DrawFunction( file_name );
      delete f;
   }
   MPIBarrier();
}
//--------------------------------------------------------------------------
void mMPIMesh2D :: DrawSubdomains( tnlGrid2D* phi,
                                  const m_char* file_name )
{
   if( MeshSize() == 1 ) return;
   m_int num = node_y_pos * 10 + node_x_pos;
   m_char file[ 1024 ];
   FileNumberEnding( file,
                     file_name,
                     num,
                     2,
                     0 );
   phi -> DrawFunction( file );
}
//--------------------------------------------------------------------------
MPI_Comm mMPIMesh2D :: GetMeshComm() const
{
   return mesh_comm;
}
//--------------------------------------------------------------------------
m_int FindNodeByCoord( MPI_Comm mesh_comm, m_int x_pos, m_int y_pos )
{
   m_int nproc;
#ifdef HAVE_MPI_H
   MPI_Comm_size( mesh_comm, &nproc );
   m_int i, coords[ 2 ];
   for( i = 0; i < nproc; i ++ )
   {
      MPI_Cart_coords( mesh_comm, i, 2, coords ); 
      if( coords[ 0 ] == x_pos && coords[ 1 ] == y_pos )
         return i;
   }
#endif
   return -1;
}
//--------------------------------------------------------------------------
m_bool CreateMesh( const mMPIMesh2D& mpi_mesh,
                   const mObjectContainer* phi_cont,
                   mObjectContainer* sub_phi_cont )
{
   dbgFunctionName( "mMPIMesh2D", "CreateMesh" );
   if( mpi_mesh. MeshSize() == 1 ) return true;
   m_int size;
   if( mpi_mesh. NodeRank() == 0 ) size = phi_cont -> Size();
   MPIBcast< m_int >( &size, 0 );
   m_int i;
   for( i = 0; i < size; i ++ )
   {
      tnlGrid2D* phi( 0 );
      if( mpi_mesh. NodeRank() == 0 )
      {
         tnlObject* obj = ( *phi_cont )[ i ];
         assert( obj -> GetType() == "tnlGrid2D" );
         phi = ( tnlGrid2D* ) obj;
      }
      tnlGrid2D* sub_phi;
      mpi_mesh. CreateMesh( phi, sub_phi );
      sub_phi_cont -> Append( sub_phi );
   }
   return true;
}
//--------------------------------------------------------------------------
void CreateGlobalGrid( const mMPIMesh2D* mpi_mesh,
                       mObjectContainer* phi_cont,
                       const mObjectContainer* sub_phi_cont )
{
   dbgFunctionName( "mMPIMesh2D", "CreateGlobalGrid" );
   if( mpi_mesh -> MeshSize() == 1 ) return;
   m_int size = sub_phi_cont -> Size();
   m_int i;
   for( i = 0; i < size; i ++ )
   {
      tnlGrid2D *phi( 0 ), *sub_phi( 0 );
      tnlObject* obj = ( *sub_phi_cont )[ i ];
      assert( obj -> GetType() == "tnlGrid2D" );
      sub_phi = ( tnlGrid2D* ) obj;
      mpi_mesh -> CreateGlobalGrid( phi, sub_phi );
      phi_cont -> Append( phi );
   }
}
//--------------------------------------------------------------------------
void Scatter( const mMPIMesh2D* mpi_mesh,
              const mObjectContainer* phi_cont,
              mObjectContainer* sub_phi_cont )
{
   m_int size;
   if( mpi_mesh -> MeshSize() == 1 ) return;
   if( mpi_mesh -> NodeRank() == 0 ) size = phi_cont -> Size();
   MPIBcast< m_int >( &size, 0 );
   m_int i;
   for( i = 0; i < size; i ++ )
   {
      tnlGrid2D* phi;
      tnlString name;
      if( mpi_mesh -> NodeRank() == 0 )
      {
         tnlObject* obj;
         obj = ( * phi_cont )[ i ];
         assert( obj -> GetType() == "tnlGrid2D" );
         phi = ( tnlGrid2D* ) obj;
         name = phi -> GetName();
      }
      name. MPIBcast( 0 );
      tnlObject* obj = sub_phi_cont -> GetObject( name. Data() );
      assert( obj );
      assert( obj -> GetType() == "tnlGrid2D" );
      tnlGrid2D* sub_phi = ( tnlGrid2D* ) obj;
      mpi_mesh -> Scatter( phi, sub_phi );
   }
}
//--------------------------------------------------------------------------
void Gather( const mMPIMesh2D* mpi_mesh,
             mObjectContainer* phi_cont,
             const mObjectContainer* sub_phi_cont )
{
   dbgFunctionName( "mMPIMesh2D", "Gather" );
   dbgExpr( mpi_mesh -> NodeRank() );
   if( mpi_mesh -> MeshSize() == 1 ) return;
   m_int size;
   if( mpi_mesh -> NodeRank() == 0 ) size = phi_cont -> Size();
   MPIBcast< m_int >( &size, 0 );
   m_int i;
   for( i = 0; i < size; i ++ )
   {
      tnlGrid2D* phi;
      tnlString name;
      if( mpi_mesh -> NodeRank() == 0 )
      {
         tnlObject* obj;
         obj = ( * phi_cont )[ i ];
         assert( obj -> GetType() == "tnlGrid2D" );
         phi = ( tnlGrid2D* ) obj;
         name = phi -> GetName();
      }
      name. MPIBcast( 0 );
      const tnlObject* obj = sub_phi_cont -> GetObject( name. Data() );
      assert( obj );
      assert( obj -> GetType() == "tnlGrid2D" );
      const tnlGrid2D* sub_phi = ( tnlGrid2D* ) obj;
      mpi_mesh -> Gather( phi, sub_phi );
   }
}
//--------------------------------------------------------------------------

