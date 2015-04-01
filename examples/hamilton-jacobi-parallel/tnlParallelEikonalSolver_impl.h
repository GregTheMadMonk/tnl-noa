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

template< typename SchemeHost, typename SchemeDevice, typename Device>
tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::tnlParallelEikonalSolver()
{
	cout << "a" << endl;
	this->device = tnlCudaDevice;

#ifdef HAVE_CUDA
	if(this->device == tnlCudaDevice)
	{
	run_host = true;
	}
#endif

	cout << "b" << endl;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::test()
{
/*
	for(int i =0; i < this->subgridValues.getSize(); i++ )
	{
		insertSubgrid(getSubgrid(i), i);
	}
*/
}

template< typename SchemeHost, typename SchemeDevice, typename Device>

bool tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::init( const tnlParameterContainer& parameters )
{
	cout << "Initializating solver..." << endl;
	const tnlString& meshLocation = parameters.getParameter <tnlString>("mesh");
	this->mesh.load( meshLocation );

	this->n = parameters.getParameter <int>("subgrid-size");
	cout << "Setting N to " << this->n << endl;

	this->subMesh.setDimensions( this->n, this->n );
	this->subMesh.setDomain( tnlStaticVector<2,double>(0.0, 0.0),
							 tnlStaticVector<2,double>(this->mesh.getHx()*(double)(this->n), this->mesh.getHy()*(double)(this->n)) );

	this->subMesh.save("submesh.tnl");

	const tnlString& initialCondition = parameters.getParameter <tnlString>("initial-condition");
	this->u0.load( initialCondition );

	//cout << this->mesh.getCellCenter(0) << endl;

	this->delta = parameters.getParameter <double>("delta");
	this->delta *= this->mesh.getHx()*this->mesh.getHy();

	cout << "Setting delta to " << this->delta << endl;

	this->tau0 = parameters.getParameter <double>("initial-tau");
	cout << "Setting initial tau to " << this->tau0 << endl;
	this->stopTime = parameters.getParameter <double>("stop-time");

	this->cflCondition = parameters.getParameter <double>("cfl-condition");
	this -> cflCondition *= sqrt(this->mesh.getHx()*this->mesh.getHy());
	cout << "Setting CFL to " << this->cflCondition << endl;

	stretchGrid();
	this->stopTime /= (double)(this->gridCols);
	this->stopTime *= (1.0+1.0/((double)(this->n) - 1.0));
	cout << "Setting stopping time to " << this->stopTime << endl;
	this->stopTime = 1.5*((double)(this->n))*parameters.getParameter <double>("stop-time")*this->mesh.getHx();
	cout << "Setting stopping time to " << this->stopTime << endl;

	cout << "Initializating scheme..." << endl;
	if(!this->schemeHost.init(parameters))
	{
		cerr << "SchemeHost failed to initialize." << endl;
		return false;
	}
	cout << "Scheme initialized." << endl;

	test();

	VectorType* tmp = new VectorType[subgridValues.getSize()];
	bool containsCurve = false;

#ifdef HAVE_CUDA

	if(this->device == tnlCudaDevice)
	{
	/*cout << "Testing... " << endl;
	if(this->device == tnlCudaDevice)
	{
	if( !initCUDA(parameters, gridRows, gridCols) )
		return false;
	}*/
		cout << "s" << endl;
	cudaMalloc(&(this->cudaSolver), sizeof(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >));
	cout << "s" << endl;
	cudaMemcpy(this->cudaSolver, this,sizeof(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >), cudaMemcpyHostToDevice);
	cout << "s" << endl;
	double** tmpdev = NULL;
	cudaMalloc(&tmpdev, sizeof(double*));
	double* tmpw;
	cudaMalloc(&tmpw, this->work_u.getSize()*sizeof(double));
	initCUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<1,1>>>(this->cudaSolver, tmpw);
	cudaDeviceSynchronize();
	cout << "s " << endl;
	//cudaMalloc(&(cudaSolver->work_u_cuda), this->work_u.getSize()*sizeof(double));
	double* tmpu = NULL;

	cudaMemcpy(&tmpu, tmpdev,sizeof(double*), cudaMemcpyDeviceToHost);
	printf("%p %p \n",tmpu,tmpw);
	cudaMemcpy(tmpw, this->work_u.getData(), this->work_u.getSize()*sizeof(double), cudaMemcpyHostToDevice);
	cout << "s "<< endl;
	}
#endif

	if(this->device == tnlHostDevice)
	{
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
	}
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
		cout << "pre 1 kernel" << endl;
		dim3 threadsPerBlock(this->n, this->n);
		dim3 numBlocks(this->gridCols,this->gridRows);
		cudaDeviceSynchronize();
		initRunCUDA<SchemeTypeHost,SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock,this->n*this->n*sizeof(double)>>>(this->cudaSolver);
		cudaDeviceSynchronize();
		cout << "post 1 kernel" << endl;
	}
#endif


	this->currentStep = 1;
	if(this->device == tnlHostDevice)
		synchronize();
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
		cudaDeviceSynchronize();
		synchronizeCUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<1,1>>>(this->cudaSolver);
	}
#endif
	cout << "Solver initialized." << endl;

	return true;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::run()
{
	if(this->device == tnlHostDevice)
	{

	bool end = false;
	while ((this->boundaryConditions.max() > 0 ) || !end)
	{
		if(this->boundaryConditions.max() == 0 )
			end=true;
		else
			end=false;
#ifdef HAVE_OPENMP
#pragma omp parallel for num_threads(3) schedule(dynamic)
#endif
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
	}
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
		bool end_cuda = false;
		dim3 threadsPerBlock(this->n, this->n);
		dim3 numBlocks(this->gridCols,this->gridRows);
		cudaDeviceSynchronize();
		cudaMalloc(&runcuda,sizeof(bool));
		cudaMemcpy(runcuda, &run_host, sizeof(bool), cudaMemcpyHostToDevice);
		bool* tmpb;
		while (run_host || !end_cuda)
		{
			cout << "a" << endl;
			if(this->boundaryConditions.max() == 0 )
				end_cuda = true;
			else
				end_cuda = false;
			cout << "a" << endl;
			cudaDeviceSynchronize();
			runCUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock,this->n*this->n*sizeof(double)>>>(this->cudaSolver);
			cout << "a" << endl;
			cudaDeviceSynchronize();
			synchronizeCUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<1,1>>>(this->cudaSolver);

			cout << "a" << endl;
			run_host = false;
			cout << "in kernel loop" << run_host << endl;
			cudaMemcpy(tmpb, &(cudaSolver->runcuda),sizeof(double*), cudaMemcpyHostToDevice);
			cudaMemcpy(&run_host, tmpb,sizeof(bool), cudaMemcpyDeviceToHost);
			cout << "in kernel loop" << run_host << endl;
		}
		cout << "b" << endl;

		double* tmpu;
		cudaMemcpy(tmpu, &(cudaSolver->work_u_cuda),sizeof(double*), cudaMemcpyHostToDevice);
		cudaMemcpy(this->work_u.getData(), tmpu, this->work_u.getSize()*sizeof(double), cudaMemcpyDeviceToHost);

	}
#endif
	contractGrid();
	this->u0.save("u-00001.tnl");
	cout << "Maximum number of calculations on one subgrid was " << this->calculationsCount.absMax() << endl;
	cout << "Average number of calculations on one subgrid was " << ( (double) this->calculationsCount.sum() / (double) this->calculationsCount.getSize() ) << endl;
	cout << "Solver finished" << endl;

}

//north - 1, east - 2, west - 4, south - 8
template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::synchronize() //needs fix ---- maybe not anymore --- but frankly: yeah, it does -- aaaa-and maybe fixed now
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


template< typename SchemeHost, typename SchemeDevice, typename Device>
int tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getOwner(int i) const
{

	return (i / (this->gridCols*this->n*this->n))*this->gridCols + (i % (this->gridCols*this->n))/this->n;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
int tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getSubgridValue( int i ) const
{
	return this->subgridValues[i];
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::setSubgridValue(int i, int value)
{
	this->subgridValues[i] = value;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
int tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getBoundaryCondition( int i ) const
{
	return this->boundaryConditions[i];
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::setBoundaryCondition(int i, int value)
{
	this->boundaryConditions[i] = value;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::stretchGrid()
{
	cout << "Stretching grid..." << endl;


	this->gridCols = ceil( ((double)(this->mesh.getDimensions().x()-1)) / ((double)(this->n-1)) );
	this->gridRows = ceil( ((double)(this->mesh.getDimensions().y()-1)) / ((double)(this->n-1)) );

	//this->gridCols = (this->mesh.getDimensions().x()-1) / (this->n-1) ;
	//this->gridRows = (this->mesh.getDimensions().y()-1) / (this->n-1) ;

	cout << "Setting gridCols to " << this->gridCols << "." << endl;
	cout << "Setting gridRows to " << this->gridRows << "." << endl;

	this->subgridValues.setSize(this->gridCols*this->gridRows);
	this->subgridValues.setValue(0);
	this->boundaryConditions.setSize(this->gridCols*this->gridRows);
	this->boundaryConditions.setValue(0);
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
	int idealStretch =this->mesh.getDimensions().x() + (this->mesh.getDimensions().x()-2)/(this->n-1);
	cout << idealStretch << endl;

	for(int i = 0; i < stretchedSize; i++)
	{
		this->unusedCell[i] = 1;
		int diff =(this->n*this->gridCols) - idealStretch ;
		//cout << "diff = " << diff <<endl;
		int k = i/this->n - i/(this->n*this->gridCols) + this->mesh.getDimensions().x()*(i/(this->n*this->n*this->gridCols)) + (i/(this->n*this->gridCols))*diff;

		if(i%(this->n*this->gridCols) - idealStretch  >= 0)
		{
			//cout << i%(this->n*this->gridCols) - idealStretch +1 << endl;
			k+= i%(this->n*this->gridCols) - idealStretch +1 ;
		}

		if(i/(this->n*this->gridCols) - idealStretch + 1  > 0)
		{
			//cout << i/(this->n*this->gridCols) - idealStretch + 1  << endl;
			k+= (i/(this->n*this->gridCols) - idealStretch +1 )* this->mesh.getDimensions().x() ;
		}

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

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::contractGrid()
{
	cout << "Contracting grid..." << endl;
	int stretchedSize = this->n*this->n*this->gridCols*this->gridRows;

	int idealStretch =this->mesh.getDimensions().x() + (this->mesh.getDimensions().x()-2)/(this->n-1);
	cout << idealStretch << endl;

	for(int i = 0; i < stretchedSize; i++)
	{
		int diff =(this->n*this->gridCols) - idealStretch ;
		int k = i/this->n - i/(this->n*this->gridCols) + this->mesh.getDimensions().x()*(i/(this->n*this->n*this->gridCols)) + (i/(this->n*this->gridCols))*diff;

		if((i%(this->n*this->gridCols) - idealStretch  < 0) && (i/(this->n*this->gridCols) - idealStretch + 1  <= 0))
		{
			//cout << i <<" : " <<i-k<< endl;
			this->u0[i-k] = this->work_u[i];
		}

	}

	cout << "Grid contracted" << endl;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
typename tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::VectorType
tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getSubgrid( const int i ) const
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

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::insertSubgrid( VectorType u, const int i )
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

template< typename SchemeHost, typename SchemeDevice, typename Device>
typename tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::VectorType
tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::runSubgrid( int boundaryCondition, VectorType u, int subGridID)
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
    	  fu[ i ] = schemeHost.getValue( this->subMesh, i, this->subMesh.getCellCoordinates(i), u, time, boundaryCondition );
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
   	}
	return solution;
}


#ifdef HAVE_CUDA


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getSubgridCUDA( const int i ,tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller, double* a)
{
	int j = threadIdx.x + threadIdx.y * blockDim.x;
	int th = (i / caller->gridCols) * caller->n*caller->n*caller->gridCols
            + (i % caller->gridCols) * caller->n
            + (j/caller->n) * caller->n*caller->gridCols
            + (j % caller->n);
	printf(" tX: %d, tY : %d, bDx: %d, Total: %d, Index: %d, pntr: %d, pntr2 %d \n", threadIdx.x,threadIdx.y,blockDim.x, j, th,caller->work_u_cuda,a);
	*a = caller->work_u_cuda[th];
	//printf("Hi %f \n", *a);
	//return ret;
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::insertSubgridCUDA( double u, const int i )
{

	int j = threadIdx.x + threadIdx.y * blockDim.x;

		int index = (i / this->gridCols)*this->n*this->n*this->gridCols + (i % this->gridCols)*this->n
					+ (j/this->n)*this->n*this->gridCols + (j % this->n);

		if( (fabs(this->work_u_cuda[index]) > fabs(u)) || (this->unusedCell_cuda[index] == 1) )
		{
			this->work_u_cuda[index] = u;
			this->unusedCell_cuda[index] = 0;
		}

}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::runSubgridCUDA( int boundaryCondition, double* u, int subGridID)
{

	__shared__ bool tmp;
	int i = threadIdx.x;
	int j = threadIdx.y;
	int l = threadIdx.y * blockDim.x + threadIdx.x;

	if(l == 0)
		tmp = false;

	__syncthreads();

	if(u[0]*u[l] <= 0.0)
		atomicOr((int*) &tmp,(int) true);



	__shared__ double value;
	if(l == 0)
		value = 0.0;

	for(int o = 0; o < blockDim.x*blockDim.y; o++)
	{
		if(l == o)
			value=Max(value,fabs(u[l]));
		__syncthreads();
	}
	//atomicMax(&value,fabs(u[l]))

	if(l == 0)
		value *= Sign(u[0]);



	__syncthreads();
	if(tmp)
	{}
	//north - 1, east - 2, west - 4, south - 8
	else if(boundaryCondition == 4)
	{
		if(j > 0)
				u[i*this->n + j] = value;
	}
	else if(boundaryCondition == 2)
	{
		if(j < this->n - 1)
			u[i*this->n + j] = value;
	}
	else if(boundaryCondition == 1)
	{
		if(i < this->n - 1)
			u[i*this->n + j] = value;
	}
	else if(boundaryCondition == 8)
	{
		if(i > 0)
			u[i*this->n + j] = value;
	}

   double time = 0.0;
   __shared__ double currentTau;
   __shared__ double maxResidue;
   double fu = 0.0;
   if(threadIdx.x * threadIdx.y == 0)
   {
	   currentTau = this->tau0;
	   maxResidue = 10.0 * this->subMesh.getHx();
   }
   double finalTime = this->stopTime;
   if( time + currentTau > finalTime ) currentTau = finalTime - time;

   __syncthreads();

   while( time < finalTime )
   {
      /****
       * Compute the RHS
       */


    	  fu = schemeDevice.getValue( this->subMesh, l, this->subMesh.getCellCoordinates(i), u, time, boundaryCondition );
      __syncthreads();


      //atomicMax(&maxResidue,fabs(fu));//maxResidue = fu. absMax();
  	for(int o = 0; o < blockDim.x*blockDim.y; o++)
  	{
  		if(l == o)
  			maxResidue=Max(maxResidue,fabs(fu));
  		__syncthreads();
  	}


      if(threadIdx.x * threadIdx.y == 0)
      {
    	  if( this -> cflCondition * maxResidue != 0.0)
    		  currentTau =  this -> cflCondition / maxResidue;

    	  if(currentTau > 1.0 * this->subMesh.getHx())
    	  {
    		  currentTau = 1.0 * this->subMesh.getHx();
    	  }

    	  if( time + currentTau > finalTime ) currentTau = finalTime - time;
    	  maxResidue = 0.0;
      }
 //
      double tau2 = 0.0;
      if((u[l]+currentTau * fu)*u[l] < 0.0 && fu != 0.0 && u[l] != 0.0 )
    	  tau2 = fabs(u[l]/(2.0*fu));

      //atomicMin(&currentTau, tau2);
    	for(int o = 0; o < blockDim.x*blockDim.y; o++)
    	{
    		if(l == o)
    			currentTau=Min(currentTau,tau2);
    		__syncthreads();
    	}


//
      __syncthreads();
      u[l] += currentTau * fu;


      time += currentTau;
   }

}

template< typename SchemeHost, typename SchemeDevice, typename Device >
__global__
void /*tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::*/runCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller)
{
	extern __shared__ double u[];
	int i = blockIdx.y * gridDim.x + blockIdx.x;
	int l = threadIdx.y * blockDim.x + threadIdx.x;

	if(caller->getSubgridValueCUDA(i) != INT_MAX)
	{
		double a;
		caller->getSubgridCUDA(i,caller, &a);
		u[l] = a;
		int bound = caller->getBoundaryConditionCUDA(i);

		if(bound & 1)
		{
			caller->runSubgridCUDA(1,u,i);
			//this->calculationsCount[i]++;
		}
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);
		if(bound & 2)
		{
			caller->runSubgridCUDA(2,u,i);
			//this->calculationsCount[i]++;
		}
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);
		if(bound & 4)
		{
			caller->runSubgridCUDA(4,u,i);
			//this->calculationsCount[i]++;
		}
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);
		if(bound & 8)
		{
			caller->runSubgridCUDA(8,u,i);
			//this->calculationsCount[i]++;
		}
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);

		if( ((bound & 2) ))
		{
			caller->runSubgridCUDA(3,u,i);
		}
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);
		if( ((bound & 4) ))
		{
			caller->runSubgridCUDA(5,u,i);
		}
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);
		if( ((bound & 2) ))
		{
			caller->runSubgridCUDA(10,u,i);
		}
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);
		if(   (bound & 4) )
		{
			caller->runSubgridCUDA(12,u,i);
		}
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);


		caller->setBoundaryConditionCUDA(i, 0);
		caller->setSubgridValueCUDA(i, caller->getSubgridValueCUDA(i) - 1 );


	}
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getOwnerCUDA(int i) const
{

	return (i / (this->gridCols*this->n*this->n))*this->gridCols
			+ (i % (this->gridCols*this->n))/this->n;
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getSubgridValueCUDA( int i ) const
{
	return this->subgridValues_cuda[i];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::setSubgridValueCUDA(int i, int value)
{
	this->subgridValues_cuda[i] = value;
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getBoundaryConditionCUDA( int i ) const
{
	return this->boundaryConditions_cuda[i];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::setBoundaryConditionCUDA(int i, int value)
{
	this->boundaryConditions_cuda[i] = value;
}



//north - 1, east - 2, west - 4, south - 8

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void /*tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::*/synchronizeCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver) //needs fix ---- maybe not anymore --- but frankly: yeah, it does -- aaaa-and maybe fixed now
{
	//cout << "Synchronizig..." << endl;
	int tmp1, tmp2;
	int grid1, grid2;

	if(cudaSolver->currentStep & 1)
	{
		for(int j = 0; j < cudaSolver->gridRows - 1; j++)
		{
			for (int i = 0; i < cudaSolver->gridCols*cudaSolver->n; i++)
			{
				tmp1 = cudaSolver->gridCols*cudaSolver->n*((cudaSolver->n-1)+j*cudaSolver->n) + i;
				tmp2 = cudaSolver->gridCols*cudaSolver->n*((cudaSolver->n)+j*cudaSolver->n) + i;
				grid1 = cudaSolver->getSubgridValueCUDA(cudaSolver->getOwnerCUDA(tmp1));
				grid2 = cudaSolver->getSubgridValueCUDA(cudaSolver->getOwnerCUDA(tmp2));

				if ((fabs(cudaSolver->work_u_cuda[tmp1]) < fabs(cudaSolver->work_u_cuda[tmp2]) - cudaSolver->delta || grid2 == INT_MAX || grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
				{
					cudaSolver->work_u_cuda[tmp2] = cudaSolver->work_u_cuda[tmp1];
					cudaSolver->unusedCell_cuda[tmp2] = 0;
					if(grid2 == INT_MAX)
					{
						cudaSolver->setSubgridValueCUDA(cudaSolver->getOwnerCUDA(tmp2), -INT_MAX);
					}
					if(! (cudaSolver->getBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp2)) & 8) )
						cudaSolver->setBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp2), cudaSolver->getBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp2))+8);
				}
				else if ((fabs(cudaSolver->work_u_cuda[tmp1]) > fabs(cudaSolver->work_u_cuda[tmp2]) + cudaSolver->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
				{
					cudaSolver->work_u_cuda[tmp1] = cudaSolver->work_u_cuda[tmp2];
					cudaSolver->unusedCell_cuda[tmp1] = 0;
					if(grid1 == INT_MAX)
					{
						cudaSolver->setSubgridValueCUDA(cudaSolver->getOwnerCUDA(tmp1), -INT_MAX);
					}
					if(! (cudaSolver->getBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp1)) & 1) )
						cudaSolver->setBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp1), cudaSolver->getBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp1))+1);
				}
			}
		}

	}
	else
	{
		for(int i = 1; i < cudaSolver->gridCols; i++)
		{
			for (int j = 0; j < cudaSolver->gridRows*cudaSolver->n; j++)
			{
				tmp1 = cudaSolver->gridCols*cudaSolver->n*j + i*cudaSolver->n - 1;
				tmp2 = cudaSolver->gridCols*cudaSolver->n*j + i*cudaSolver->n ;
				grid1 = cudaSolver->getSubgridValueCUDA(cudaSolver->getOwnerCUDA(tmp1));
				grid2 = cudaSolver->getSubgridValueCUDA(cudaSolver->getOwnerCUDA(tmp2));

				if ((fabs(cudaSolver->work_u_cuda[tmp1]) < fabs(cudaSolver->work_u_cuda[tmp2]) - cudaSolver->delta || grid2 == INT_MAX || grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
				{
					cudaSolver->work_u_cuda[tmp2] = cudaSolver->work_u_cuda[tmp1];
					cudaSolver->unusedCell_cuda[tmp2] = 0;
					if(grid2 == INT_MAX)
					{
						cudaSolver->setSubgridValueCUDA(cudaSolver->getOwnerCUDA(tmp2), -INT_MAX);
					}
					if(! (cudaSolver->getBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp2)) & 4) )
						cudaSolver->setBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp2), cudaSolver->getBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp2))+4);
				}
				else if ((fabs(cudaSolver->work_u_cuda[tmp1]) > fabs(cudaSolver->work_u_cuda[tmp2]) + cudaSolver->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
				{
					cudaSolver->work_u_cuda[tmp1] = cudaSolver->work_u_cuda[tmp2];
					cudaSolver->unusedCell_cuda[tmp1] = 0;
					if(grid1 == INT_MAX)
					{
						cudaSolver->setSubgridValueCUDA(cudaSolver->getOwnerCUDA(tmp1), -INT_MAX);
					}
					if(! (cudaSolver->getBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp1)) & 2) )
						cudaSolver->setBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp1), cudaSolver->getBoundaryConditionCUDA(cudaSolver->getOwnerCUDA(tmp1))+2);
				}
			}
		}
	}


	cudaSolver->currentStep++;
	int stepValue = cudaSolver->currentStep + 4;
	for (int i = 0; i < cudaSolver->subgridValues.getSize(); i++)
	{
		if( cudaSolver->getSubgridValueCUDA(i) == -INT_MAX )
			cudaSolver->setSubgridValueCUDA(i, stepValue);
	}

	int maxi = 0;
	for(int q=0; q < cudaSolver->n*cudaSolver->n;q++)
		maxi=Max(maxi,cudaSolver->boundaryConditions_cuda[q]);

	*(cudaSolver->runcuda) = (maxi > 0);
	//cout << "Grid synchronized at step " << (this->currentStep - 1 ) << endl;

}








template< typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void /*tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::*/initCUDA( tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver, double* ptr )
{
	//cout << "Initializating solver..." << endl;
	//const tnlString& meshLocation = parameters.getParameter <tnlString>("mesh");
	//this->mesh_cuda.load( meshLocation );

	//this->n_cuda = parameters.getParameter <int>("subgrid-size");
	//cout << "Setting N << this->n_cuda << endl;

	//this->subMesh_cuda.setDimensions( this->n_cuda, this->n_cuda );
	//this->subMesh_cuda.setDomain( tnlStaticVector<2,double>(0.0, 0.0),
							 //tnlStaticVector<2,double>(this->mesh_cuda.getHx()*(double)(this->n_cuda), this->mesh_cuda.getHy()*(double)(this->n_cuda)) );

	//this->subMesh_cuda.save("submesh.tnl");

//	const tnlString& initialCondition = parameters.getParameter <tnlString>("initial-condition");
//	this->u0.load( initialCondition );

	//cout << this->mesh.getCellCenter(0) << endl;

	//this->delta_cuda = parameters.getParameter <double>("delta");
	//this->delta_cuda *= this->mesh_cuda.getHx()*this->mesh_cuda.getHy();

	//cout << "Setting delta to " << this->delta << endl;

	//this->tau0_cuda = parameters.getParameter <double>("initial-tau");
	//cout << "Setting initial tau to " << this->tau0_cuda << endl;
	//this->stopTime_cuda = parameters.getParameter <double>("stop-time");

	//this->cflCondition_cuda = parameters.getParameter <double>("cfl-condition");
	//this -> cflCondition_cuda *= sqrt(this->mesh_cuda.getHx()*this->mesh_cuda.getHy());
	//cout << "Setting CFL to " << this->cflCondition << endl;
////
////

//	this->gridRows_cuda = gridRows;
//	this->gridCols_cuda = gridCols;

	cudaSolver->work_u_cuda = ptr;//(double*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*cudaSolver->n*cudaSolver->n*sizeof(double));
	cudaSolver->unusedCell_cuda = (int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*cudaSolver->n*cudaSolver->n*sizeof(int));
	cudaSolver->subgridValues_cuda =(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*sizeof(int));
	cudaSolver->boundaryConditions_cuda =(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*sizeof(int));
	//cudaMemcpy(ptr,&(cudaSolver->work_u_cuda), sizeof(double*),cudaMemcpyDeviceToHost);
	ptr = cudaSolver->work_u_cuda;
	printf("GPU memory allocated. %p \n",cudaSolver->work_u_cuda);
	//cudaSolver->work_u_cuda[50] = 32.153438;
////
////
	//stretchGrid();
	//this->stopTime_cuda /= (double)(this->gridCols_cuda);
	//this->stopTime_cuda *= (1.0+1.0/((double)(this->n_cuda) - 1.0));
	//cout << "Setting stopping time to " << this->stopTime << endl;
	//this->stopTime_cuda = 1.5*((double)(this->n_cuda))*parameters.getParameter <double>("stop-time")*this->mesh_cuda.getHx();
	//cout << "Setting stopping time to " << this->stopTime << endl;

	//cout << "Initializating scheme..." << endl;
	//if(!this->schemeDevice.init(parameters))
//	{
		//cerr << "Scheme failed to initialize." << endl;
//		return false;
//	}
	//cout << "Scheme initialized." << endl;

	//test();

//	this->currentStep_cuda = 1;
	//return true;
}




//extern __shared__ double array[];
template< typename SchemeHost, typename SchemeDevice, typename Device >
__global__
void /*tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::*/initRunCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller)

{


	extern __shared__ double u[];
	//printf("%p\n",caller->work_u_cuda);

	int i = blockIdx.y * gridDim.x + blockIdx.x;
	int l = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int containsCurve;
	if(l == 0)
		containsCurve = 0;

	double a;
	caller->getSubgridCUDA(i,caller, &a);
	printf("%f   %f\n",a , u[l]);
	u[l] = a;
	printf("Hi %f \n", u[l]);
	__syncthreads();
	printf("hurewrwr %f \n", u[l]);
	if(u[0] * u[l] <= 0.0)
	{
		printf("000 \n");
		atomicMax( &containsCurve, 1);
	}

	__syncthreads();
	printf("hu");
	if(containsCurve)
	{
		caller->runSubgridCUDA(0,u,i);
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);
		caller->setSubgridValueCUDA(i, 4);
	}
	printf("das");

}

#endif /*HAVE_CUDA*/

#endif /* TNLPARALLELEIKONALSOLVER_IMPL_H_ */
