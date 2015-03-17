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
#include <core/mfilename.h>

template< typename Scheme>
tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::tnlParallelEikonalSolver()
{
}

template< typename Scheme>
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::test()
{
/*
	for(int i =0; i < this->subgridValues.getSize(); i++ )
	{
		insertSubgrid(getSubgrid(i), i);
	}
*/
}

template< typename Scheme>
bool tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::init( const tnlParameterContainer& parameters )
{
	cout << "Initializating solver..." << endl;
	const tnlString& meshLocation = parameters.GetParameter <tnlString>("mesh");
	this->mesh.load( meshLocation );

	this->n = parameters.GetParameter <int>("subgrid-size");
	cout << "Setting N to " << this->n << endl;

	this->subMesh.setDimensions( this->n, this->n );
	this->subMesh.setDomain( tnlStaticVector<2,double>(0.0, 0.0),
							 tnlStaticVector<2,double>(this->mesh.getHx()*(double)(this->n), this->mesh.getHy()*(double)(this->n)) );

	this->subMesh.save("submesh.tnl");

	const tnlString& initialCondition = parameters.GetParameter <tnlString>("initial-condition");
	this->u0.load( initialCondition );

	//cout << this->mesh.getCellCenter(0) << endl;

	this->delta = parameters.GetParameter <double>("delta");
	this->delta *= this->mesh.getHx()*this->mesh.getHy();

	cout << "Setting delta to " << this->delta << endl;

	this->tau0 = parameters.GetParameter <double>("initial-tau");
	cout << "Setting initial tau to " << this->tau0 << endl;
	this->stopTime = parameters.GetParameter <double>("stop-time");

	this->cflCondition = parameters.GetParameter <double>("cfl-condition");
	this -> cflCondition *= sqrt(this->mesh.getHx()*this->mesh.getHy());
	cout << "Setting CFL to " << this->cflCondition << endl;

	stretchGrid();
	this->stopTime /= (double)(this->gridCols);
	this->stopTime *= (1.0+1.0/((double)(this->n) - 1.0));
	cout << "Setting stopping time to " << this->stopTime << endl;
	this->stopTime = ((double)(this->n) - 1.0)*parameters.GetParameter <double>("stop-time")*this->mesh.getHx();
	cout << "Setting stopping time to " << this->stopTime << endl;

	cout << "Initializating scheme..." << endl;
	if(!this->scheme.init(parameters))
	{
		cerr << "Scheme failed to initialize." << endl;
		return false;
	}
	cout << "Scheme initialized." << endl;

	test();

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
				j=tmp[i].getSize();
			}

		}
		if(containsCurve)
		{
			//cout << "Computing initial SDF on subgrid " << i << "." << endl;
			insertSubgrid(runSubgrid(0, tmp[i],i), i);
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
	bool end = false;
	while ((this->boundaryConditions.max() > 0 ) || !end)
	{
		if(this->boundaryConditions.max() == 0 )
			end=true;
		else
			end=false;
#pragma omp parallel for num_threads(3) schedule(dynamic)
		for(int i = 0; i < this->subgridValues.getSize(); i++)
		{
			if(getSubgridValue(i) != INT_MAX)
			{
				//cout << "subMesh: " << i << ", BC: " << getBoundaryCondition(i) << endl;

				if(getBoundaryCondition(i) & 1)
				{
					insertSubgrid( runSubgrid(1, getSubgrid(i),i), i);
					this->calculationsCount[i]++;
				}
				if(getBoundaryCondition(i) & 2)
				{
					insertSubgrid( runSubgrid(2, getSubgrid(i),i), i);
					this->calculationsCount[i]++;
				}
				if(getBoundaryCondition(i) & 4)
				{
					insertSubgrid( runSubgrid(4, getSubgrid(i),i), i);
					this->calculationsCount[i]++;
				}
				if(getBoundaryCondition(i) & 8)
				{
					insertSubgrid( runSubgrid(8, getSubgrid(i),i), i);
					this->calculationsCount[i]++;
				}


				if( ((getBoundaryCondition(i) & 2) )//|| (getBoundaryCondition(i) & 1))
					/*	&&(!(getBoundaryCondition(i) & 5) && !(getBoundaryCondition(i) & 10)) */)
				{
					//cout << "3 @ " << getBoundaryCondition(i) << endl;
					insertSubgrid( runSubgrid(3, getSubgrid(i),i), i);
				}
				if( ((getBoundaryCondition(i) & 4) )//|| (getBoundaryCondition(i) & 1))
					/*	&&(!(getBoundaryCondition(i) & 3) && !(getBoundaryCondition(i) & 12)) */)
				{
					//cout << "5 @ " << getBoundaryCondition(i) << endl;
					insertSubgrid( runSubgrid(5, getSubgrid(i),i), i);
				}
				if( ((getBoundaryCondition(i) & 2) )//|| (getBoundaryCondition(i) & 8))
					/*	&&(!(getBoundaryCondition(i) & 12) && !(getBoundaryCondition(i) & 3))*/ )
				{
					//cout << "10 @ " << getBoundaryCondition(i) << endl;
					insertSubgrid( runSubgrid(10, getSubgrid(i),i), i);
				}
				if(   ((getBoundaryCondition(i) & 4) )//|| (getBoundaryCondition(i) & 8))
					/*&&(!(getBoundaryCondition(i) & 10) && !(getBoundaryCondition(i) & 5)) */)
				{
					//cout << "12 @ " << getBoundaryCondition(i) << endl;
					insertSubgrid( runSubgrid(12, getSubgrid(i),i), i);
				}


				/*if(getBoundaryCondition(i))
				{
					insertSubgrid( runSubgrid(15, getSubgrid(i),i), i);
				}*/

				setBoundaryCondition(i, 0);

				setSubgridValue(i, getSubgridValue(i)-1);

			}
		}
		synchronize();
	}


	contractGrid();
	this->u0.save("u-00001.tnl");
	cout << "Maximum number of calculations on one subgrid was " << this->calculationsCount.absMax() << endl;
	cout << "Average number of calculations on one subgrid was " << ( (double) this->calculationsCount.sum() / (double) this->calculationsCount.getSize() ) << endl;
	cout << "Solver finished" << endl;

}

//north - 1, east - 2, west - 4, south - 8
template< typename Scheme >
void tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::synchronize() //needs fix ---- maybe not anymore --- but frankly: yeah, it does -- aaaa-and maybe fixed now
{
	cout << "Synchronizig..." << endl;
	int tmp1, tmp2;
	int grid1, grid2;

	if(this->currentStep & 1)
	{
		for(int j = 0; j < this->gridRows - 1; j++)
		{
			for (int i = 0; i < this->gridCols*this->n; i++)
			{
				tmp1 = this->gridCols*this->n*((this->n-1)+j*this->n) + i;
				tmp2 = this->gridCols*this->n*((this->n)+j*this->n) + i;
				grid1 = getSubgridValue(getOwner(tmp1));
				grid2 = getSubgridValue(getOwner(tmp2));
				if(getOwner(tmp1)==getOwner(tmp2))
					cout << "i, j" << i << "," << j << endl;
				if ((fabs(this->work_u[tmp1]) < fabs(this->work_u[tmp2]) - this->delta || grid2 == INT_MAX || grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
				{
					this->work_u[tmp2] = this->work_u[tmp1];
					this->unusedCell[tmp2] = 0;
					if(grid2 == INT_MAX)
					{
						setSubgridValue(getOwner(tmp2), -INT_MAX);
					}
					if(! (getBoundaryCondition(getOwner(tmp2)) & 8) )
						setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+8);
				}
				else if ((fabs(this->work_u[tmp1]) > fabs(this->work_u[tmp2]) + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
				{
					this->work_u[tmp1] = this->work_u[tmp2];
					this->unusedCell[tmp1] = 0;
					if(grid1 == INT_MAX)
					{
						setSubgridValue(getOwner(tmp1), -INT_MAX);
					}
					if(! (getBoundaryCondition(getOwner(tmp1)) & 1) )
						setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+1);
				}
			}
		}

	}
	else
	{
		for(int i = 1; i < this->gridCols; i++)
		{
			for (int j = 0; j < this->gridRows*this->n; j++)
			{
				tmp1 = this->gridCols*this->n*j + i*this->n - 1;
				tmp2 = this->gridCols*this->n*j + i*this->n ;
				grid1 = getSubgridValue(getOwner(tmp1));
				grid2 = getSubgridValue(getOwner(tmp2));
				if(getOwner(tmp1)==getOwner(tmp2))
					cout << "i, j" << i << "," << j << endl;
				if ((fabs(this->work_u[tmp1]) < fabs(this->work_u[tmp2]) - this->delta || grid2 == INT_MAX || grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
				{
					this->work_u[tmp2] = this->work_u[tmp1];
					this->unusedCell[tmp2] = 0;
					if(grid2 == INT_MAX)
					{
						setSubgridValue(getOwner(tmp2), -INT_MAX);
					}
					if(! (getBoundaryCondition(getOwner(tmp2)) & 4) )
						setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+4);
				}
				else if ((fabs(this->work_u[tmp1]) > fabs(this->work_u[tmp2]) + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
				{
					this->work_u[tmp1] = this->work_u[tmp2];
					this->unusedCell[tmp1] = 0;
					if(grid1 == INT_MAX)
					{
						setSubgridValue(getOwner(tmp1), -INT_MAX);
					}
					if(! (getBoundaryCondition(getOwner(tmp1)) & 2) )
						setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+2);
				}
			}
		}
	}


	this->currentStep++;
	int stepValue = this->currentStep + 4;
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


	//this->gridCols = ceil( ((double)(this->mesh.getDimensions().x()-1)) / ((double)(this->n-1)) );
	//this->gridRows = ceil( ((double)(this->mesh.getDimensions().y()-1)) / ((double)(this->n-1)) );

	this->gridCols = (this->mesh.getDimensions().x()-1) / (this->n-1) ;
	this->gridRows = (this->mesh.getDimensions().y()-1) / (this->n-1) ;

	cout << "Setting gridCols to " << this->gridCols << "." << endl;
	cout << "Setting gridRows to " << this->gridRows << "." << endl;

	this->subgridValues.setSize(this->gridCols*this->gridRows);
	this->boundaryConditions.setSize(this->gridCols*this->gridRows);
	this->calculationsCount.setSize(this->gridCols*this->gridRows);
	this->calculationsCount.setValue(0);

	for(int i = 0; i < this->subgridValues.getSize(); i++ )
	{
		this->subgridValues[i] = INT_MAX;
		this->boundaryConditions[i] = 0;
	}

	int stretchedSize = this->n*this->n*this->gridCols*this->gridRows;

	if(!this->work_u.setSize(stretchedSize))
		cerr << "Could not allocate memory for stretched grid." << endl;
	if(!this->unusedCell.setSize(stretchedSize))
		cerr << "Could not allocate memory for supporting stretched grid." << endl;

	for(int i = 0; i < stretchedSize; i++)
	{
		this->unusedCell[i] = 1;
		int k = i/this->n - i/(this->n*this->gridCols) + this->mesh.getDimensions().x()*(i/(this->n*this->n*this->gridCols));
		//cout << "i = " << i << " : i-k = " << i-k << endl;
		/*int j=(i % (this->n*this->gridCols)) - ( (this->mesh.getDimensions().x() - this->n)/(this->n - 1) + this->mesh.getDimensions().x() - 1)
				+ (this->n*this->gridCols - this->mesh.getDimensions().x())*(i/(this->n*this->n*this->gridCols)) ;

		if(j > 0)
			k += j;

		int l = i-k - (this->u0.getSize() - 1);
		int m = (l % this->mesh.getDimensions().x());

		if(l>0)
			k+= l + ( (l / this->mesh.getDimensions().x()) + 1 )*this->mesh.getDimensions().x() - (l % this->mesh.getDimensions().x());*/

		this->work_u[i] = this->u0[i-k];
		//cout << (i-k) <<endl;
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
		int k = i/this->n - i/(this->n*this->gridCols) + this->mesh.getDimensions().x()*(i/(this->n*this->n*this->gridCols));
		/*int j=(i % (this->n*this->gridCols)) - ( (this->mesh.getDimensions().x() - this->n)/(this->n - 1) + this->mesh.getDimensions().x() - 1)
				+ (this->n*this->gridCols - this->mesh.getDimensions().x())*(i/(this->n*this->n*this->gridCols)) ;
		int l = i-k - (this->u0.getSize() - 1);

		if(!(j > 0) && !(l>0))*/
			this->u0[i-k] = this->work_u[i];

	}

	cout << "Grid contracted" << endl;
}

template< typename Scheme >
typename tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::VectorType
tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::getSubgrid( const int i ) const
{
	VectorType u;
	u.setSize(this->n*this->n);

	for( int j = 0; j < u.getSize(); j++)
	{
		u[j] = this->work_u[ (i / this->gridCols) * this->n*this->n*this->gridCols
		                     + (i % this->gridCols) * this->n
		                     + (j/this->n) * this->n*this->gridCols
		                     + (j % this->n) ];
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
		if( (fabs(this->work_u[index]) > fabs(u[j])) || (this->unusedCell[index] == 1) )
		{
			this->work_u[index] = u[j];
			this->unusedCell[index] = 0;
		}
		//OMP UNLOCK index
	}
}

template< typename Scheme >
typename tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::VectorType
tnlParallelEikonalSolver<Scheme, double, tnlHost, int>::runSubgrid( int boundaryCondition, VectorType u, int subGridID)
{

	VectorType fu;

	fu.setLike(u);
	fu.setValue( 0.0 );

/*
 *          Insert Euler-Solver Here
 */

	/**/

	/*for(int i = 0; i < u.getSize(); i++)
	{
		int x = this->subMesh.getCellCoordinates(i).x();
		int y = this->subMesh.getCellCoordinates(i).y();

		if(x == 0 && (boundaryCondition & 4) && y ==0)
		{
			if((u[subMesh.getCellYSuccessor( i )] - u[i])/subMesh.getHy() > 1.0)
			{
				//cout << "x = 0; y = 0" << endl;
				u[i] = u[subMesh.getCellYSuccessor( i )] - subMesh.getHy();
			}
		}
		else if(x == 0 && (boundaryCondition & 4) && y == subMesh.getDimensions().y() - 1)
		{
			if((u[subMesh.getCellYPredecessor( i )] - u[i])/subMesh.getHy() > 1.0)
			{
				//cout << "x = 0; y = n" << endl;
				u[i] = u[subMesh.getCellYPredecessor( i )] - subMesh.getHy();
			}
		}


		else if(x == subMesh.getDimensions().x() - 1 && (boundaryCondition & 2) && y ==0)
		{
			if((u[subMesh.getCellYSuccessor( i )] - u[i])/subMesh.getHy() > 1.0)
			{
				//cout << "x = n; y = 0" << endl;
				u[i] = u[subMesh.getCellYSuccessor( i )] - subMesh.getHy();
			}
		}
		else if(x == subMesh.getDimensions().x() - 1 && (boundaryCondition & 2) && y == subMesh.getDimensions().y() - 1)
		{
			if((u[subMesh.getCellYPredecessor( i )] - u[i])/subMesh.getHy() > 1.0)
			{
				//cout << "x = n; y = n" << endl;
				u[i] = u[subMesh.getCellYPredecessor( i )] - subMesh.getHy();
			}
		}


		else if(y == 0 && (boundaryCondition & 8) && x ==0)
		{
			if((u[subMesh.getCellXSuccessor( i )] - u[i])/subMesh.getHx() > 1.0)
			{
				//cout << "y = 0; x = 0" << endl;
				u[i] = u[subMesh.getCellXSuccessor( i )] - subMesh.getHx();
			}
		}
		else if(y == 0 && (boundaryCondition & 8) && x == subMesh.getDimensions().x() - 1)
		{
			if((u[subMesh.getCellXPredecessor( i )] - u[i])/subMesh.getHx() > 1.0)
			{
				//cout << "y = 0; x = n" << endl;
				u[i] = u[subMesh.getCellXPredecessor( i )] - subMesh.getHx();
			}
		}


		else if(y == subMesh.getDimensions().y() - 1 && (boundaryCondition & 1) && x ==0)
		{
			if((u[subMesh.getCellXSuccessor( i )] - u[i])/subMesh.getHx() > 1.0)			{
				//cout << "y = n; x = 0" << endl;
				u[i] = u[subMesh.getCellXSuccessor( i )] - subMesh.getHx();
			}
		}
		else if(y == subMesh.getDimensions().y() - 1 && (boundaryCondition & 1) && x == subMesh.getDimensions().x() - 1)
		{
			if((u[subMesh.getCellXPredecessor( i )] - u[i])/subMesh.getHx() > 1.0)
			{
				//cout << "y = n; x = n" << endl;
				u[i] = u[subMesh.getCellXPredecessor( i )] - subMesh.getHx();
			}
		}
	}*/

	/**/


/*	bool tmp = false;
	for(int i = 0; i < u.getSize(); i++)
	{
		if(u[0]*u[i] <= 0.0)
			tmp=true;
	}


	if(tmp)
	{}
	else if(boundaryCondition == 4)
	{
		int i;
		for(i = 0; i < u.getSize() - subMesh.getDimensions().x() ; i=subMesh.getCellYSuccessor(i))
		{
			int j;
			for(j = i; j < subMesh.getDimensions().x() - 1; j=subMesh.getCellXSuccessor(j))
			{
				u[j] = u[i];
			}
			u[j] = u[i];
		}
		int j;
		for(j = i; j < subMesh.getDimensions().x() - 1; j=subMesh.getCellXSuccessor(j))
		{
			u[j] = u[i];
		}
		u[j] = u[i];
	}
	else if(boundaryCondition == 8)
	{
		int i;
		for(i = 0; i < subMesh.getDimensions().x() - 1; i=subMesh.getCellXSuccessor(i))
		{
			int j;
			for(j = i; j < u.getSize() - subMesh.getDimensions().x(); j=subMesh.getCellYSuccessor(j))
			{
				u[j] = u[i];
			}
			u[j] = u[i];
		}
		int j;
		for(j = i; j < u.getSize() - subMesh.getDimensions().x(); j=subMesh.getCellYSuccessor(j))
		{
			u[j] = u[i];
		}
		u[j] = u[i];

	}
	else if(boundaryCondition == 2)
	{
		int i;
		for(i = subMesh.getDimensions().x() - 1; i < u.getSize() - subMesh.getDimensions().x() ; i=subMesh.getCellYSuccessor(i))
		{
			int j;
			for(j = i; j > (i-1)*subMesh.getDimensions().x(); j=subMesh.getCellXPredecessor(j))
			{
				u[j] = u[i];
			}
			u[j] = u[i];
		}
		int j;
		for(j = i; j > (i-1)*subMesh.getDimensions().x(); j=subMesh.getCellXPredecessor(j))
		{
			u[j] = u[i];
		}
		u[j] = u[i];
	}
	else if(boundaryCondition == 1)
	{
		int i;
		for(i = (subMesh.getDimensions().y() - 1)*subMesh.getDimensions().x(); i < u.getSize() - 1; i=subMesh.getCellXSuccessor(i))
		{
			int j;
			for(j = i; j >=subMesh.getDimensions().x(); j=subMesh.getCellYPredecessor(j))
			{
				u[j] = u[i];
			}
			u[j] = u[i];
		}
		int j;
		for(j = i; j >=subMesh.getDimensions().x(); j=subMesh.getCellYPredecessor(j))
		{
			u[j] = u[i];
		}
		u[j] = u[i];
	}
*/
	/**/



	bool tmp = false;
	for(int i = 0; i < u.getSize(); i++)
	{
		if(u[0]*u[i] <= 0.0)
			tmp=true;
	}
	//if(this->currentStep + 3 < getSubgridValue(subGridID))
		//tmp = true;


	double value = Sign(u[0]) * u.absMax();

	if(tmp)
	{}


	//north - 1, east - 2, west - 4, south - 8
	else if(boundaryCondition == 4)
	{
		for(int i = 0; i < this->n; i++)
			for(int j = 1;j < this->n; j++)
				//if(fabs(u[i*this->n + j]) <  fabs(u[i*this->n]))
				u[i*this->n + j] = value;// u[i*this->n];
	}
	else if(boundaryCondition == 2)
	{
		for(int i = 0; i < this->n; i++)
			for(int j =0 ;j < this->n -1; j++)
				//if(fabs(u[i*this->n + j]) < fabs(u[(i+1)*this->n - 1]))
				u[i*this->n + j] = value;// u[(i+1)*this->n - 1];
	}
	else if(boundaryCondition == 1)
	{
		for(int j = 0; j < this->n; j++)
			for(int i = 0;i < this->n - 1; i++)
				//if(fabs(u[i*this->n + j]) < fabs(u[j + this->n*(this->n - 1)]))
				u[i*this->n + j] = value;// u[j + this->n*(this->n - 1)];
	}
	else if(boundaryCondition == 8)
	{
		for(int j = 0; j < this->n; j++)
			for(int i = 1;i < this->n; i++)
				//if(fabs(u[i*this->n + j]) < fabs(u[j]))
				u[i*this->n + j] = value;// u[j];
	}

/*

	else if(boundaryCondition == 5)
	{
		for(int i = 0; i < this->n - 1; i++)
			for(int j = 1;j < this->n; j++)
				//if(fabs(u[i*this->n + j]) <  fabs(u[i*this->n]))
				u[i*this->n + j] = value;// u[i*this->n];
	}
	else if(boundaryCondition == 10)
	{
		for(int i = 1; i < this->n; i++)
			for(int j =0 ;j < this->n -1; j++)
				//if(fabs(u[i*this->n + j]) < fabs(u[(i+1)*this->n - 1]))
				u[i*this->n + j] = value;// u[(i+1)*this->n - 1];
	}
	else if(boundaryCondition == 3)
	{
		for(int j = 0; j < this->n - 1; j++)
			for(int i = 0;i < this->n - 1; i++)
				//if(fabs(u[i*this->n + j]) < fabs(u[j + this->n*(this->n - 1)]))
				u[i*this->n + j] = value;// u[j + this->n*(this->n - 1)];
	}
	else if(boundaryCondition == 12)
	{
		for(int j = 1; j < this->n; j++)
			for(int i = 1;i < this->n; i++)
				//if(fabs(u[i*this->n + j]) < fabs(u[j]))
				u[i*this->n + j] = value;// u[j];
	}

*/

	/**/

	/*if (u.max() > 0.0)
		this->stopTime *=(double) this->gridCols;*/


   double time = 0.0;
   double currentTau = this->tau0;
   double finalTime = this->stopTime;// + 3.0*(u.max() - u.min());
   if( time + currentTau > finalTime ) currentTau = finalTime - time;

   double maxResidue( 1.0 );
   //double lastResidue( 10000.0 );
   while( time < finalTime /*|| maxResidue > subMesh.getHx()*/)
   {
      /****
       * Compute the RHS
       */

      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  fu[ i ] = scheme.getValue( this->subMesh, i, this->subMesh.getCellCoordinates(i), u, time, boundaryCondition );
      }
      maxResidue = fu. absMax();


      if( this -> cflCondition * maxResidue != 0.0)
    	  currentTau =  this -> cflCondition / maxResidue;

     /* if (maxResidue < 0.05)
    	  cout << "Max < 0.05" << endl;*/
      if(currentTau > 1.0 * this->subMesh.getHx())
      {
    	  //cout << currentTau << " >= " << 2.0 * this->subMesh.getHx() << endl;
    	  currentTau = 1.0 * this->subMesh.getHx();
      }
      /*if(maxResidue > lastResidue)
    	  currentTau *=(1.0/10.0);*/


      if( time + currentTau > finalTime ) currentTau = finalTime - time;
      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  //cout << "Too big RHS! i = " << i << ", fu = " << fu[i] << ", u = " << u[i] << endl;
    	  if((u[i]+currentTau * fu[ i ])*u[i] < 0.0 && fu[i] != 0.0 && u[i] != 0.0 )
    		  currentTau = fabs(u[i]/(2.0*fu[i]));
    	  
      }


      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  double add = u[i] + currentTau * fu[ i ];
    	  //if( fabs(u[i]) < fabs(add) or (this->subgridValues[subGridID] == this->currentStep +4) )
    		  u[ i ] = add;
      }
      time += currentTau;

      //cout << '\r' << flush;
     //cout << maxResidue << "   " << currentTau << " @ " << time << flush;
     //lastResidue = maxResidue;
   }
   //cout << "Time: " << time << ", Res: " << maxResidue <<endl;
	/*if (u.max() > 0.0)
		this->stopTime /=(double) this->gridCols;*/

	VectorType solution;
	solution.setLike(u);
    for( int i = 0; i < u.getSize(); i ++ )
  	{
		solution[i]=u[i];
   	}
	return solution;
}


#endif /* TNLPARALLELEIKONALSOLVER_IMPL_H_ */
