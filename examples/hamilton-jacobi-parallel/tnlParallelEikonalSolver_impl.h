/***************************************************************************
                          tnlParallelEikonalSolver_impl.h  -  description
                             -------------------
    begin                : Nov 28 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLPARALLELEIKONALSOLVER_IMPL_H_
#define TNLPARALLELEIKONALSOLVER_IMPL_H_


#include "tnlParallelEikonalSolver.h"

template< typename Scheme>
tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::tnlParallelEikonalSolver()
{
}

template< typename Scheme>
bool tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::init( const tnlParameterContainer& parameters )
{
	cout << "Initializating solver..." << endl;
	const tnlString& meshLocation = parameters.GetParameter <tnlString>("mesh");
	this->mesh.load( meshLocation );

	this->n = parameters.GetParameter <int>("subgrid-size");

	this->subMesh.setDimensions( this->n, this->n );
	this->subMesh.setDomain( tnlStaticVector<2,double>(0.0, 0.0),
							 tnlStaticVector<2,double>(this->mesh.getHx()*(double)(this->n), this->mesh.getHy()*(double)(this->n)) );


	const tnlString& initialCondition = parameters.GetParameter <tnlString>("initial-condition");
	this->u0.load( initialCondition );

	this->delta = parameters.GetParameter <double>("delta");
	this->delta *= sqrt(this->mesh.getHx()*this->mesh.getHx() + this->mesh.getHy()*this->mesh.getHy());

	this->tau0 = parameters.GetParameter <double>("initial-tau");
	this->stopTime = parameters.GetParameter <double>("stop-time");

	this->cflCondition = parameters.GetParameter <double>("cfl-condition");

	stretchGrid();
	this->stopTime /= (double)(this->gridCols);

	cout << "Initializating scheme..." << endl;
	if(!this->scheme.init(parameters))
	{
		cerr << "Scheme failed to initialize." << endl;
		return false;
	}
	cout << "Scheme initialized." << endl;


	VectorType* tmp = new VectorType[subgridValues.getSize()];
	bool containsCurve = false;

	for(int i = 0; i < this->subgridValues.getSize(); i++)
	{

		if(! tmp[i].setSize(this->n * this->n))
			cout << "Could not allocate tmp["<< i <<"] array." << endl;
			tmp[i] = getSubgrid(i);
		containsCurve = false;

		for(int j = 0; j < tmp[i].getSize(); j++)
		{
			if(tmp[i][0]*tmp[i][j] <= 0.0)
			{
				containsCurve = true;
				//cout << tmp[i][0] << ":" << tmp[i][j] << endl;
				j=tmp[i].getSize();
			}

		}
		if(containsCurve)
		{
			//cout << "Computing initial SDF on subgrid " << i << "." << endl;
			insertSubgrid(runSubgrid(0, tmp[i]), i);
			setSubgridValue(i, 4);
			//cout << "Computed initial SDF on subgrid " << i  << "." << endl;
		}
		containsCurve = false;

	}

	this->currentStep = 1;
	synchronize();
	cout << "Solver initialized." << endl;
	return true;
}

template< typename Scheme >
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::run()
{
	while (this->subgridValues.max() > 0)
	{

		for(int i = 0; i < this->subgridValues.getSize(); i++)
		{
			if(getSubgridValue(i) != INT_MAX && getSubgridValue(i) > 0)
			{
				if(getBoundaryCondition(i) & 1)
				{
					insertSubgrid( runSubgrid(1, getSubgrid(i)), i);
					setBoundaryCondition(i, getBoundaryCondition(i) - 1);
				}
				if(getBoundaryCondition(i) & 2)
				{
					insertSubgrid( runSubgrid(2, getSubgrid(i)), i);
					setBoundaryCondition(i, getBoundaryCondition(i) - 2);
				}
				if(getBoundaryCondition(i) & 4)
				{
					insertSubgrid( runSubgrid(4, getSubgrid(i)), i);
					setBoundaryCondition(i, getBoundaryCondition(i) - 4);
				}
				if(getBoundaryCondition(i) & 8)
				{
					insertSubgrid( runSubgrid(8, getSubgrid(i)), i);
					setBoundaryCondition(i, getBoundaryCondition(i) - 8);
				}

				setSubgridValue(i, getSubgridValue(i)-1);
			}
		}

		synchronize();
	}

	contractGrid();
	this->u0.save("u-00001.tnl");
	cout << "Solver finished" << endl;

}
//north - 1, east - 2, west - 4, south - 8
template< typename Scheme >
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::synchronize()
{
	cout << "Synchronizig..." << endl;
	int tmp1, tmp2;
	int grid1, grid2;

	for(int j = 0; j < this->gridRows - 1; j++)
	{
		for (int i = 0; i < this->gridCols*this->n; i++)
		{
			tmp1 = this->gridCols*this->n*((this->n-1)+j*this->n) + i;
			tmp2 = this->gridCols*this->n*((this->n)+j*this->n) + i;
			grid1 = getSubgridValue(getOwner(tmp1));
			grid2 = getSubgridValue(getOwner(tmp2));
			if ((this->work_u[tmp1] < this->work_u[tmp2] - this->delta || grid2 == INT_MAX|| grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
			{
				this->work_u[tmp2] = this->work_u[tmp1];
				if(getSubgridValue(getOwner(tmp2)) == INT_MAX)
				{
					setSubgridValue(getOwner(tmp2), -INT_MAX);
				}
				setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+1);
			}
			else if ((this->work_u[tmp1] > this->work_u[tmp2] + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
			{
				this->work_u[tmp1] = this->work_u[tmp2];
				if(getSubgridValue(getOwner(tmp1)) == INT_MAX)
				{
					setSubgridValue(getOwner(tmp1), -INT_MAX);
				}
				setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+8);
			}
			/*else if ( (grid1 != INT_MAX && grid1 != -INT_MAX) || (grid2 != INT_MAX && grid2 != -INT_MAX))
			{
				if(getSubgridValue(getOwner(tmp2)) == INT_MAX)
				{
					setSubgridValue(getOwner(tmp2), -INT_MAX);
				}
				else if(getSubgridValue(getOwner(tmp1)) == INT_MAX)
				{
					setSubgridValue(getOwner(tmp1), -INT_MAX);
				}
			}*/
		}

	}


	for(int i = 1; i < this->gridCols; i++)
	{
		for (int j = 0; j < this->gridRows*this->n; j++)
		{
			tmp1 = this->gridCols*this->n*j + i*this->n;
			tmp2 = this->gridCols*this->n*j + i*this->n + 1;
			if ((this->work_u[tmp1] < this->work_u[tmp2] - this->delta || grid2 == INT_MAX|| grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
			{
				this->work_u[tmp2] = this->work_u[tmp1];
				if(getSubgridValue(getOwner(tmp2)) == INT_MAX)
				{
					setSubgridValue(getOwner(tmp2), -INT_MAX);
				}
				setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+2);
			}
			else if ((this->work_u[tmp1] > this->work_u[tmp2] + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
			{
				this->work_u[tmp1] = this->work_u[tmp2];
				if(getSubgridValue(getOwner(tmp1)) == INT_MAX)
				{
					setSubgridValue(getOwner(tmp1), -INT_MAX);
				}
				setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+4);
			}
			/*else if ( (grid1 != INT_MAX && grid1 != -INT_MAX) || (grid2 != INT_MAX && grid2 != -INT_MAX))
			{
				if(getSubgridValue(getOwner(tmp2)) == INT_MAX)
				{
					setSubgridValue(getOwner(tmp2), -INT_MAX);
				}
				else if(getSubgridValue(getOwner(tmp1)) == INT_MAX)
				{
					setSubgridValue(getOwner(tmp1), -INT_MAX);
				}
			}*/
		}

	}

	int stepValue = this->currentStep + 3;
	this->currentStep++;
	for (int i = 0; i < this->subgridValues.getSize(); i++)
	{
		if( getSubgridValue(i) == -INT_MAX )
			setSubgridValue(i, stepValue);
	}

	cout << "Grid synchronized at step " << (this->currentStep - 1 ) << endl;

}


template< typename Scheme >
int tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::getOwner(int i) const
{

	return (i / (this->gridCols*this->n*this->n))*this->gridCols + (i % (this->gridCols*this->n))/this->n;
}

template< typename Scheme >
int tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::getSubgridValue( int i ) const
{
	return this->subgridValues[i];
}

template< typename Scheme >
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::setSubgridValue(int i, int value)
{
	this->subgridValues[i] = value;
}

template< typename Scheme >
int tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::getBoundaryCondition( int i ) const
{
	return this->boundaryConditions[i];
}

template< typename Scheme >
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::setBoundaryCondition(int i, int value)
{
	this->boundaryConditions[i] = value;
}

template< typename Scheme >
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::stretchGrid()
{
	cout << "Stretching grid..." << endl;


	this->gridCols = ceil((double)(this->mesh.getDimensions().x())/(double)(this->n));
	this->gridRows = ceil((double)(this->mesh.getDimensions().y())/(double)(this->n));

	this->subgridValues.setSize(this->gridCols*this->gridRows);
	this->boundaryConditions.setSize(this->gridCols*this->gridRows);

	for(int i = 0; i < this->subgridValues.getSize(); i++ )
	{
		this->subgridValues[i] = INT_MAX;
		this->boundaryConditions[i] = 0;
	}

	int stretchedSize = this->n*this->n*this->gridCols*this->gridRows;

	if(!this->work_u.setSize(stretchedSize))
		cerr << "Could not allocate memory for stretched grid." << endl;


	for(int i = 0; i < stretchedSize; i++)
	{
		int k = i/this->n - i/(this->n*this->gridCols) + this->n*this->gridCols*(i/(this->n*this->n*this->gridCols));
		int j=(i % (this->n*this->gridCols)) - ( (this->mesh.getDimensions().x() - this->n)/(this->n - 1) + this->mesh.getDimensions().x());

		if(j > 0)
			k += j;

		this->work_u[i] = this->u0[i-k];
	}

	cout << "Grid stretched." << endl;
}

template< typename Scheme >
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::contractGrid()
{
	cout << "Contracting grid..." << endl;
	int stretchedSize = this->n*this->n*this->gridCols*this->gridRows;

	for(int i = 0; i < stretchedSize; i++)
	{
		int k = i/this->n - i/(this->n*this->gridCols) + this->n*this->gridCols*(i/(this->n*this->n*this->gridCols));
		int j=(i % (this->n*this->gridCols)) - ((this->mesh.getDimensions().x() - this->n)/(this->n - 1) + this->mesh.getDimensions().x());

		if(!(j > 0))
			this->u0[i-k] = this->work_u[i];
	}

	//MISSING FILL FOR BOTTOM
	cout << "Grid contracted" << endl;
}

template< typename Scheme >
typename tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::VectorType
tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::getSubgrid( const int i ) const
{
	VectorType u;
	u.setSize(this->n*this->n);

	for( int j = 0; j < this->n*this->n; j++)
	{
		u[j] = this->work_u[(i / this->gridCols)*this->n*this->n*this->gridCols + (i % this->gridCols)*this->n + (j/this->n)*this->n*this->gridCols + (j % this->n)];
	}
	return u;
}

template< typename Scheme >
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::insertSubgrid( VectorType u, const int i )
{
	for( int j = 0; j < this->n*this->n; j++)
	{
		int index = (i / this->gridCols)*this->n*this->n*this->gridCols + (i % this->gridCols)*this->n + (j/this->n)*this->n*this->gridCols + (j % this->n);
		//OMP LOCK index
		if(this->work_u[index] > u[j])
			this->work_u[index] = u[j];
		//OMP UNLOCK index
	}
}

template< typename Scheme >
typename tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::VectorType
tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::runSubgrid( int boundaryCondition, VectorType u)
{
	VectorType fu;

	fu.setLike(u);
	fu.setValue( 0.0 );

/*
 *          Insert Euler-Solver Here
 */

   double time = 0.0;
   double currentTau = this->tau0;
   if( time + currentTau > this -> stopTime ) currentTau = this -> stopTime - time;
   /*this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   this -> refreshSolverMonitor();*/

   /****
    * Start the main loop
    */
   while( time < this->stopTime )
   {
      /****
       * Compute the RHS
       */
      //this->problem->getExplicitRHS( time, currentTau, u, fu );

      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  fu[ i ] = scheme.getValue( this->subMesh, i, this->subMesh.getCellCoordinates(i), u, time, boundaryCondition );
      }


      //RealType lastResidue = this->getResidue();
      double maxResidue( 0.0 );
      //cout << fu. absMax() << endl;
      if( this -> cflCondition != 0.0 )
      {
         maxResidue = fu. absMax();
         if( currentTau * maxResidue > this -> cflCondition )
         {
            currentTau *= 0.9;
            continue;
         }
      }
      //RealType newResidue( 0.0 );
      //computeNewTimeLevel( u, currentTau, newResidue );
      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  double add = currentTau * fu[ i ];
    	  u[ i ] += add;
          //localResidue += fabs( add );
      }
      //this->setResidue( newResidue );

      /****
       * When time is close to stopTime the new residue
       * may be inaccurate significantly.
       */
      //if( currentTau + time == this -> stopTime ) this->setResidue( lastResidue );
      time += currentTau;

      /*if( ! this->nextIteration() )
         return false;*/

      /****
       * Compute the new time step.
       */
      if( this -> cflCondition != 0.0 )
         currentTau /= 0.95;

      if( time + currentTau > this -> stopTime )
         currentTau = this -> stopTime - time; //we don't want to keep such tau
      //else this -> tau = currentTau;

      //this -> refreshSolverMonitor();

      /****
       * Check stop conditions.
       */
      /*if( time >= this->getStopTime()  ||
          ( this -> getConvergenceResidue() != 0.0 && this->getResidue() < this -> getConvergenceResidue() ) )
      {
         //this -> refreshSolverMonitor();
         return true;
      }*/


   }


	VectorType solution;
	solution.setLike(u);
    for( int i = 0; i < u.getSize(); i ++ )
  	{
		solution[i]=u[i];
   	}

	return solution;
}


#endif /* TNLPARALLELEIKONALSOLVER_IMPL_H_ */
