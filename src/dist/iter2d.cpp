/***************************************************************************
                          iter2d.cpp  -  description
                             -------------------
    begin                : 2005/08/09
    copyright            : (C) 2005 by Tom� Oberhuber
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

#include <diff/mGrid2D.h>
#include <debug/tnlDebug.h>
#include "iter1d.h"
#include "mDistAux.h"

/*void mDstIter2D( const m_real& t,
                 mGrid2D** _phi,
                 mGrid2D** _f_phi,
                 void* _aux )
{
   dbgFunctionName( "", "f_2D" );
         
   assert( _aux );
   
   mGrid2D& phi = * _phi[ 0 ];
   mGrid2D& f_phi = * _f_phi[ 0 ];

   mDistAux< mGrid2D >* aux = ( mDistAux< mGrid2D>* ) _aux;
   mDistIterMethod method = aux -> method;
   const m_real& epsilon = aux -> epsilon;

   m_int x_size = phi. XSize();
   m_int y_size = phi. YSize();
   m_real h_x = phi. H(). x;
   m_real h_y = phi. H(). y;
   m_int i, j;

   mGrid2D& mod_grad = aux -> mod_grad;
   mGrid2D& laplace = aux -> laplace;

   if( method == mDstRegularisedScheme )
   {
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
            laplace( i, j ) = phi. Laplace( i, j );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
         {
            m_real phi_x_f = phi. Partial_x_f( i, j );
            m_real phi_x_b = phi. Partial_x_b( i, j );
            m_real phi_y_f = phi. Partial_y_f( i, j );
            m_real phi_y_b = phi. Partial_y_b( i, j );
            mod_grad( i, j ) = 0.5 * ( 
                  sqrt( phi_x_f * phi_x_f + phi_y_f * phi_y_f ) +
                  sqrt( phi_x_b * phi_x_b + phi_y_b * phi_y_b ) );
                  
         }
      //Extrapolate( laplace );
      //Extrapolate( mod_grad );
      m_real h = Min( phi. H().x , phi. H(). y );
      for( i = 0; i < x_size; i ++ )
         for( j = 0; j < y_size; j ++ )
            f_phi( i, j ) = epsilon * h * laplace( i, j ) + 
               sign( phi( i, j ) ) * ( 1.0 - mod_grad( i, j ) );
      return;
   }
 
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
      {
         m_real grxf = phi. Partial_x_f( i, j );
         m_real grxb = phi. Partial_x_b( i, j );
         m_real gryf = phi. Partial_y_f( i, j );
         m_real gryb = phi. Partial_y_b( i, j );
         m_real grxfp = Max( grxf, 0.0 );
         m_real grxfn = Min( grxf, 0.0 );
         m_real grxbp = Max( grxb, 0.0 );
         m_real grxbn = Min( grxb, 0.0 );
         m_real gryfp = Max( gryf, 0.0 );
         m_real gryfn = Min( gryf, 0.0 );
         m_real grybp = Max( gryb, 0.0 );
         m_real grybn = Min( gryb, 0.0 );
         m_real d_plus, d_minus;
         m_real maxxbf, maxybf, maxxfb, maxyfb;
         switch( method )
         {
            case mDstUpwindScheme:
                d_plus = sqrt( grxbp * grxbp + grxfn * grxfn +
                               grybp * grybp + gryfn * gryfn );
                d_minus =sqrt( grxfp * grxfp + grxbn * grxbn +
                               gryfp * gryfp + grybn * grybn );
            break;
            case mDstGodunovScheme:
                maxxbf = Max( grxbp, -1.0 * grxfn );
                maxybf = Max( grybp, -1.0 * gryfn );
                maxxfb = Max( grxfp, -1.0 * grxbn );
                maxyfb = Max( gryfp, -1.0 * grybn );
                d_plus = sqrt( maxxbf * maxxbf + maxybf * maxybf ); 
                d_minus =sqrt( maxxfb * maxxfb + maxyfb * maxyfb );
            break;
            default:
            break;
         }
         m_real F_plus = Max( sign( phi( i, j ) ), 0.0 );       
         m_real F_minus = Min( sign( phi( i, j ) ), 0.0 );
         f_phi( i, j ) = sign( phi( i, j ) ) - ( F_plus * d_plus + F_minus * d_minus );
      }
}
//--------------------------------------------------------------------------
void IterDist2D( mGrid2D* phi, mConfig& config )
{
   dbgFunctionName( "", "IterDist1D" );

   mESolver2D solver;
   
   m_real init_t = config. Get< m_real >( "init-t" );
   m_real tau = config. Get< m_real >( "init-tau" );
   m_real final_t = config. Get< m_real >( "final-t" );
   m_real epsilon = config. Get< m_real >( "epsilon" );
   m_real output_period = config. Get< m_real >( "output-period" );
   m_real adaptivity = config. Get< m_real >( "adaptivity" );
   m_int step = config. Get< m_int >( "init-step" );
   const m_char* method_name = 
      config. Get< mString >( "method-name" ). Data();
   const m_char* output_file = 
      config. Get< mString >( "output-file" ). Data();
   const m_char* output_file_ending = 
      config. Get< mString >( "output-file-ending" ). Data();
   
   m_real t( init_t );
   m_real break_t = t + output_period;
   if( ! output_period ) output_period = final_t - init_t;
   if( ! tau ) tau = output_period;
   if( ! adaptivity ) adaptivity = 1.0e-4;

   mDistIterMethod method( mDstNone );
   if( strcmp( method_name, "iter-regularised" ) == 0 )
      method = mDstRegularisedScheme;
   if( strcmp( method_name, "iter-upwind" ) == 0 )
      method = mDstUpwindScheme;
   if( strcmp( method_name, "iter-godunov" ) == 0 )
      method = mDstGodunovScheme;
   if( ! method )
   {
      cerr << "Uknown method name '" << method_name << "' !" << endl;
      return;
   }
   
   m_int x_size = phi -> XSize();
   m_int y_size = phi -> YSize();
   cout << "problem name: " << config. Get< mString >( "problem-name" ) << endl;
   cout << "Dim = 2 " << endl;
   cout << "Size = " << x_size << "x" << y_size << endl;
   cout << "h = " << phi -> H() << endl;
   cout << "init_t = " << init_t << " final_t = " << final_t << endl;
   cout << "Method name: " << method_name << endl;
   cout << "Adaptivity: " << adaptivity << endl;
   if( method == mDstRegularisedScheme )
      cout << "epsilon = " << epsilon << endl;
   cout << "Initial step " << step << endl;
   cout << "output file : " << output_file << endl
        << "output file ending : " << output_file_ending << endl;
   cout << "Starting computation" << endl;
 
   mDistAux< mGrid2D > aux( *phi, method, epsilon );
   
   solver. Init( "merson",
                 *phi,
                 2 );
   solver. SetT( t );
   //solver. SetFinalT( final_t );
   solver. SetTau( tau );
   solver. SetAdaptivity( adaptivity );
   
   m_char file_name[ 1024 ];
   if( output_file )
   {
      FileNumberEnding( file_name,
                        output_file,
                        step,
                        5,
                        output_file_ending );
      cout << endl << "Writing file: " << file_name << endl;
      assert( phi );
      phi -> DrawFunction( file_name );
   }
   solver. StartTiming();
   while( t < final_t )
   {
      
      solver. SetBreakT( break_t );
      //solver. Solve( &phi, mDstIter2D, 0, &aux );
      t = solver. GetT();
      break_t += output_period;
      step ++;
      
      if( output_file )
      {
         FileNumberEnding( file_name,
                           output_file,
                           step,
                           5,
                           output_file_ending );
         cout << endl << "Writing file: " << file_name << endl;
         assert( phi );
         phi -> DrawFunction( file_name );
      }
      
      if( output_curve_file )
      {
         FileNumberEnding( file_name,
                           output_curve_file,
                           step,
                           5,
                           file_ending );
         cout << endl << "Writing file: " << file_name << endl;
         curve. Identify( *_phi, curve_level_set );
         curve. Draw( file_name );
         curve. Delete();
      }

   }
}*/
