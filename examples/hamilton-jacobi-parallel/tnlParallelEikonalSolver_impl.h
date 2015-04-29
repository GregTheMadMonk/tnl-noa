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

//north - 1, east - 2, west - 4, south - 8

#define FROM_NORTH 1
#define FROM_EAST 2
#define FROM_WEST 4
#define FROM_SOUTH 8

#include "tnlParallelEikonalSolver.h"
#include <core/mfilename.h>

template< typename SchemeHost, typename SchemeDevice, typename Device>
tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::tnlParallelEikonalSolver()
{
	cout << "a" << endl;
	this->device = tnlHostDevice;

#ifdef HAVE_CUDA
	if(this->device == tnlCudaDevice)
	{
	run_host = 1;
	}
#endif

	cout << "b" << endl;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::test()
{
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
	cudaMalloc(&(this->cudaSolver), sizeof(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >));
	cudaMemcpy(this->cudaSolver, this,sizeof(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >), cudaMemcpyHostToDevice);
	double** tmpdev = NULL;
	cudaMalloc(&tmpdev, sizeof(double*));
	cudaMalloc(&(this->tmpw), this->work_u.getSize()*sizeof(double));
	cudaMalloc(&(this->runcuda), sizeof(int));
	cudaDeviceSynchronize();
	checkCudaDevice;
	int* tmpUC;
	cudaMalloc(&(tmpUC), this->work_u.getSize()*sizeof(int));
	cudaMemcpy(tmpUC, this->unusedCell.getData(), this->unusedCell.getSize()*sizeof(int), cudaMemcpyHostToDevice);

	initCUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<1,1>>>(this->cudaSolver, (this->tmpw), (this->runcuda),tmpUC);
	cudaDeviceSynchronize();
	checkCudaDevice;
	double* tmpu = NULL;

	cudaMemcpy(&tmpu, tmpdev,sizeof(double*), cudaMemcpyDeviceToHost);
	cudaMemcpy((this->tmpw), this->work_u.getData(), this->work_u.getSize()*sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	checkCudaDevice;
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
			insertSubgrid(runSubgrid(0, tmp[i],i), i);
			setSubgridValue(i, 4);
		}
		containsCurve = false;

	}
	}
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
		cudaDeviceSynchronize();
		checkCudaDevice;
		dim3 threadsPerBlock(this->n, this->n);
		dim3 numBlocks(this->gridCols,this->gridRows);
		cudaDeviceSynchronize();
		checkCudaDevice;
		initRunCUDA<SchemeTypeHost,SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock,3*this->n*this->n*sizeof(double)>>>(this->cudaSolver);
		cudaDeviceSynchronize();
	}
#endif


	this->currentStep = 1;
	if(this->device == tnlHostDevice)
		synchronize();
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
		dim3 threadsPerBlock(this->n, this->n);
		dim3 numBlocks(this->gridCols,this->gridRows);

		checkCudaDevice;

		synchronizeCUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock>>>(this->cudaSolver);
		cudaDeviceSynchronize();
		checkCudaDevice;
		synchronize2CUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,1>>>(this->cudaSolver);
		cudaDeviceSynchronize();
		checkCudaDevice;

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
#pragma omp parallel for num_threads(4) schedule(dynamic)
#endif
		for(int i = 0; i < this->subgridValues.getSize(); i++)
		{
			if(getSubgridValue(i) != INT_MAX)
			{

				if(getBoundaryCondition(i) & FROM_NORTH)
				{
					insertSubgrid( runSubgrid(FROM_NORTH, getSubgrid(i),i), i);
					this->calculationsCount[i]++;
				}
				if(getBoundaryCondition(i) & FROM_EAST)
				{
					insertSubgrid( runSubgrid(FROM_EAST, getSubgrid(i),i), i);
					this->calculationsCount[i]++;
				}
				if(getBoundaryCondition(i) & FROM_WEST)
				{
					insertSubgrid( runSubgrid(FROM_WEST, getSubgrid(i),i), i);
					this->calculationsCount[i]++;
				}
				if(getBoundaryCondition(i) & FROM_SOUTH)
				{
					insertSubgrid( runSubgrid(FROM_SOUTH, getSubgrid(i),i), i);
					this->calculationsCount[i]++;
				}


				if( ((getBoundaryCondition(i) & FROM_EAST) ))
				{
					insertSubgrid( runSubgrid(FROM_NORTH + FROM_EAST, getSubgrid(i),i), i);
				}
				if( ((getBoundaryCondition(i) & FROM_WEST) ))
				{
					insertSubgrid( runSubgrid(FROM_NORTH + FROM_WEST, getSubgrid(i),i), i);
				}
				if( ((getBoundaryCondition(i) & FROM_EAST) ))
				{
					insertSubgrid( runSubgrid(FROM_SOUTH + FROM_EAST, getSubgrid(i),i), i);
				}
				if(   ((getBoundaryCondition(i) & FROM_WEST) ))
				{
					insertSubgrid( runSubgrid(FROM_SOUTH + FROM_WEST, getSubgrid(i),i), i);
				}

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
		checkCudaDevice;
		bool* tmpb;
		cudaMemcpy(&(this->run_host),this->runcuda,sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		checkCudaDevice;
		int i = 1;
		time_diff = 0.0;
		while (run_host || !end_cuda)
		{
			cout << "Computing at step "<< i++ << endl;
			if(run_host != 0 )
				end_cuda = true;
			else
				end_cuda = false;
			cudaDeviceSynchronize();
			checkCudaDevice;
			start = std::clock();
			runCUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock,3*this->n*this->n*sizeof(double)>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			time_diff += (std::clock() - start) / (double)(CLOCKS_PER_SEC);

			//start = std::clock();
			synchronizeCUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			checkCudaDevice;
			synchronize2CUDA<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,1>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			checkCudaDevice;
			//time_diff += (std::clock() - start) / (double)(CLOCKS_PER_SEC);

			cudaMemcpy(&run_host, (this->runcuda),sizeof(int), cudaMemcpyDeviceToHost);
		}
		cout << "Solving time was: " << time_diff << endl;
		cudaMemcpy(this->work_u.getData(), (this->tmpw), this->work_u.getSize()*sizeof(double), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
	}
#endif
	contractGrid();
	this->u0.save("u-00001.tnl");
	cout << "Maximum number of calculations on one subgrid was " << this->calculationsCount.absMax() << endl;
	cout << "Average number of calculations on one subgrid was " << ( (double) this->calculationsCount.sum() / (double) this->calculationsCount.getSize() ) << endl;
	cout << "Solver finished" << endl;

#ifdef HAVE_CUDA
	if(this->device == tnlCudaDevice)
	{
		cudaFree(this->runcuda);
		cudaFree(this->tmpw);
		cudaFree(this->cudaSolver);
	}
#endif

}

//north - 1, east - 2, west - 4, south - 8
template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::synchronize()
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
					if(! (getBoundaryCondition(getOwner(tmp2)) & FROM_SOUTH) )
						setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+FROM_SOUTH);
				}
				else if ((fabs(this->work_u[tmp1]) > fabs(this->work_u[tmp2]) + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
				{
					this->work_u[tmp1] = this->work_u[tmp2];
					this->unusedCell[tmp1] = 0;
					if(grid1 == INT_MAX)
					{
						setSubgridValue(getOwner(tmp1), -INT_MAX);
					}
					if(! (getBoundaryCondition(getOwner(tmp1)) & FROM_NORTH) )
						setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+FROM_NORTH);
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
					if(! (getBoundaryCondition(getOwner(tmp2)) & FROM_WEST) )
						setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+FROM_WEST);
				}
				else if ((fabs(this->work_u[tmp1]) > fabs(this->work_u[tmp2]) + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
				{
					this->work_u[tmp1] = this->work_u[tmp2];
					this->unusedCell[tmp1] = 0;
					if(grid1 == INT_MAX)
					{
						setSubgridValue(getOwner(tmp1), -INT_MAX);
					}
					if(! (getBoundaryCondition(getOwner(tmp1)) & FROM_EAST) )
						setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+FROM_EAST);
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
		int k = i/this->n - i/(this->n*this->gridCols) + this->mesh.getDimensions().x()*(i/(this->n*this->n*this->gridCols)) + (i/(this->n*this->gridCols))*diff;

		if(i%(this->n*this->gridCols) - idealStretch  >= 0)
		{
			k+= i%(this->n*this->gridCols) - idealStretch +1 ;
		}

		if(i/(this->n*this->gridCols) - idealStretch + 1  > 0)
		{
			k+= (i/(this->n*this->gridCols) - idealStretch +1 )* this->mesh.getDimensions().x() ;
		}

		this->work_u[i] = this->u0[i-k];
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
	bool tmp = false;
	for(int i = 0; i < u.getSize(); i++)
	{
		if(u[0]*u[i] <= 0.0)
			tmp=true;
		int centreGID = (this->n*(subGridID / this->gridRows)+ (this->n >> 1))*(this->n*this->gridCols) + this->n*(subGridID % this->gridRows) + (this->n >> 1);
		if(this->unusedCell[centreGID] == 0 || boundaryCondition == 0)
			tmp = true;
	}

	double value = Sign(u[0]) * u.absMax();

	if(tmp)
	{}


	//north - 1, east - 2, west - 4, south - 8
	else if(boundaryCondition == FROM_WEST)
	{
		for(int i = 0; i < this->n; i++)
			for(int j = 1;j < this->n; j++)
				u[i*this->n + j] = value;// u[i*this->n];
	}
	else if(boundaryCondition == FROM_EAST)
	{
		for(int i = 0; i < this->n; i++)
			for(int j =0 ;j < this->n -1; j++)
				u[i*this->n + j] = value;// u[(i+1)*this->n - 1];
	}
	else if(boundaryCondition == FROM_NORTH)
	{
		for(int j = 0; j < this->n; j++)
			for(int i = 0;i < this->n - 1; i++)
				u[i*this->n + j] = value;// u[j + this->n*(this->n - 1)];
	}
	else if(boundaryCondition == FROM_SOUTH)
	{
		for(int j = 0; j < this->n; j++)
			for(int i = 1;i < this->n; i++)
				u[i*this->n + j] = value;// u[j];
	}

   double time = 0.0;
   double currentTau = this->tau0;
   double finalTime = this->stopTime;// + 3.0*(u.max() - u.min());
   if( time + currentTau > finalTime ) currentTau = finalTime - time;

   double maxResidue( 1.0 );
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

      if(currentTau > 1.0 * this->subMesh.getHx())
      {
    	  currentTau = 1.0 * this->subMesh.getHx();
      }


      if( time + currentTau > finalTime ) currentTau = finalTime - time;
      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  if((u[i]+currentTau * fu[ i ])*u[i] < 0.0 && fu[i] != 0.0 && u[i] != 0.0 )
    		  currentTau = fabs(u[i]/(2.0*fu[i]));
    	  
      }


      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  double add = u[i] + currentTau * fu[ i ];
    		  u[ i ] = add;
      }
      time += currentTau;

   }
	VectorType solution;
	solution.setLike(u);
    for( int i = 0; i < u.getSize(); i ++ )
  	{
    	solution[i]=u[i];
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
	*a = caller->work_u_cuda[th];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::updateSubgridCUDA( const int i ,tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller, double* a)
{
	int j = threadIdx.x + threadIdx.y * blockDim.x;
	int index = (i / caller->gridCols) * caller->n*caller->n*caller->gridCols
            + (i % caller->gridCols) * caller->n
            + (j/caller->n) * caller->n*caller->gridCols
            + (j % caller->n);

	if( (fabs(caller->work_u_cuda[index]) > fabs(*a)) || (caller->unusedCell_cuda[index] == 1) )
	{
		caller->work_u_cuda[index] = *a;
		caller->unusedCell_cuda[index] = 0;

	}

	*a = caller->work_u_cuda[index];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::insertSubgridCUDA( double u, const int i )
{


	int j = threadIdx.x + threadIdx.y * blockDim.x;

		int index = (i / this->gridCols)*this->n*this->n*this->gridCols
					+ (i % this->gridCols)*this->n
					+ (j/this->n)*this->n*this->gridCols
					+ (j % this->n);

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

	__shared__ int tmp;
	__shared__ double value;
	volatile double* sharedTau = &u[blockDim.x*blockDim.y];
	volatile double* absVal = &u[2*blockDim.x*blockDim.y];
	int i = threadIdx.x;
	int j = threadIdx.y;
	int l = threadIdx.y * blockDim.x + threadIdx.x;
	bool computeFU = !((i == 0 && (boundaryCondition & FROM_WEST)) or
			 (i == blockDim.x - 1 && (boundaryCondition & FROM_EAST)) or
			 (j == 0 && (boundaryCondition & FROM_SOUTH)) or
			 (j == blockDim.y - 1  && (boundaryCondition & FROM_NORTH)));

	if(l == 0)
	{
		tmp = 0;
		int centreGID = (blockDim.y*blockIdx.y + (blockDim.y>>1))*(blockDim.x*gridDim.x) + blockDim.x*blockIdx.x + (blockDim.x>>1);
		if(this->unusedCell_cuda[centreGID] == 0 || boundaryCondition == 0)
			tmp = 1;
	}
	__syncthreads();


	__syncthreads();
	if(tmp !=1)
	{
		if(computeFU)
			absVal[l]=0.0;
		else
			absVal[l] = fabs(u[l]);

		__syncthreads();

	      if((blockDim.x == 16) && (l < 128))		absVal[l] = Max(absVal[l],absVal[l+128]);
	      __syncthreads();
	      if((blockDim.x == 16) && (l < 64))		absVal[l] = Max(absVal[l],absVal[l+64]);
	      __syncthreads();
	      if(l < 32)    							absVal[l] = Max(absVal[l],absVal[l+32]);
	      if(l < 16)								absVal[l] = Max(absVal[l],absVal[l+16]);
	      if(l < 8)									absVal[l] = Max(absVal[l],absVal[l+8]);
	      if(l < 4)									absVal[l] = Max(absVal[l],absVal[l+4]);
	      if(l < 2)									absVal[l] = Max(absVal[l],absVal[l+2]);
	      if(l < 1)									value   = Sign(u[0])*Max(absVal[l],absVal[l+1]);
		__syncthreads();

		if(computeFU)
		{
//			u[l] = value;
			if(boundaryCondition==FROM_NORTH)
				u[l] = u[i + blockDim.x*(blockDim.y-1)];
			else if(boundaryCondition==FROM_SOUTH)
				u[l] = u[i];
			else if(boundaryCondition==FROM_EAST)
				u[l] = u[blockDim.x-1 + blockDim.x*j];
			else if(boundaryCondition==FROM_WEST)
				u[l] = u[blockDim.x*j];
		}
	}

   double time = 0.0;
   __shared__ double currentTau;
   double cfl = this->cflCondition;
   double fu = 0.0;
   if(threadIdx.x * threadIdx.y == 0)
   {
	   currentTau = this->tau0;
   }
   double finalTime = this->stopTime;
   __syncthreads();
   if( time + currentTau > finalTime ) currentTau = finalTime - time;


   while( time < finalTime )
   {

	  if(computeFU)
		  fu = schemeHost.getValueDev( this->subMesh, l, tnlStaticVector<2,int>(i,j)/*this->subMesh.getCellCoordinates(l)*/, u, time, boundaryCondition);

	  sharedTau[l]=cfl/fabs(fu);

      if(l == 0)
    	  if(sharedTau[0] > 1.0 * this->subMesh.getHx())	sharedTau[0] = 1.0 * this->subMesh.getHx();

      if(l == blockDim.x*blockDim.y - 1)
    	  if( time + sharedTau[l] > finalTime )		sharedTau[l] = finalTime - time;
      __syncthreads();


      if((blockDim.x == 16) && (l < 128))		sharedTau[l] = Min(sharedTau[l],sharedTau[l+128]);
      __syncthreads();
      if((blockDim.x == 16) && (l < 64))		sharedTau[l] = Min(sharedTau[l],sharedTau[l+64]);
      __syncthreads();
      if(l < 32)    							sharedTau[l] = Min(sharedTau[l],sharedTau[l+32]);
      if(l < 16)								sharedTau[l] = Min(sharedTau[l],sharedTau[l+16]);
      if(l < 8)									sharedTau[l] = Min(sharedTau[l],sharedTau[l+8]);
      if(l < 4)									sharedTau[l] = Min(sharedTau[l],sharedTau[l+4]);
      if(l < 2)									sharedTau[l] = Min(sharedTau[l],sharedTau[l+2]);
      if(l < 1)									currentTau   = Min(sharedTau[l],sharedTau[l+1]);
	__syncthreads();

      u[l] += currentTau * fu;
      time += currentTau;
      __syncthreads();
   }


}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int>::getOwnerCUDA(int i) const
{

	return ((i / (this->gridCols*this->n*this->n))*this->gridCols
			+ (i % (this->gridCols*this->n))/this->n);
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
void synchronizeCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver) //needs fix ---- maybe not anymore --- but frankly: yeah, it does -- aaaa-and maybe fixed now
{

	__shared__ int boundary[4]; // north,east,west,south
	__shared__ int subgridValue;
	__shared__ int newSubgridValue;


	int gid = (blockDim.y*blockIdx.y + threadIdx.y)*blockDim.x*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
	double u = cudaSolver->work_u_cuda[gid];
	double u_cmp;
	int subgridValue_cmp=INT_MAX;
	int boundary_index=0;


	if(threadIdx.x+threadIdx.y == 0)
	{
		subgridValue = cudaSolver->getSubgridValueCUDA(blockIdx.y*gridDim.x + blockIdx.x);
		boundary[0] = 0;
		boundary[1] = 0;
		boundary[2] = 0;
		boundary[3] = 0;
		newSubgridValue = 0;
	}
	__syncthreads();



	if(		(threadIdx.x == 0 					/*	&& !(cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.y == 0 				 	/*	&& (cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.x == blockDim.x - 1	 	/*	&& !(cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.y == blockDim.y - 1 		/*	&& (cudaSolver->currentStep & 1)*/) 		)
	{
		if(threadIdx.x == 0 && (blockIdx.x != 0)/* && !(cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid - 1];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA(blockIdx.y*gridDim.x + blockIdx.x - 1);
			boundary_index = 2;
		}

		if(threadIdx.x == blockDim.x - 1 && (blockIdx.x != gridDim.x - 1)/* && !(cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid + 1];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA(blockIdx.y*gridDim.x + blockIdx.x + 1);
			boundary_index = 1;
		}
		__threadfence();
		if((subgridValue == INT_MAX || fabs(u_cmp) + cudaSolver->delta < fabs(u) ) && (subgridValue_cmp != INT_MAX && subgridValue_cmp != -INT_MAX))
		{
			cudaSolver->unusedCell_cuda[gid] = 0;
			atomicMax(&newSubgridValue, INT_MAX);
			atomicMax(&boundary[boundary_index], 1);
			cudaSolver->work_u_cuda[gid] = u_cmp;
			u=u_cmp;
		}
		if(threadIdx.y == 0 && (blockIdx.y != 0)/* && (cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid - blockDim.x*gridDim.x];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA((blockIdx.y - 1)*gridDim.x + blockIdx.x);
			boundary_index = 3;
		}
		if(threadIdx.y == blockDim.y - 1 && (blockIdx.y != gridDim.y - 1)/* && (cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid + blockDim.x*gridDim.x];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA((blockIdx.y + 1)*gridDim.x + blockIdx.x);
			boundary_index = 0;
		}

		__threadfence();
		if((subgridValue == INT_MAX || fabs(u_cmp) + cudaSolver->delta < fabs(u) ) && (subgridValue_cmp != INT_MAX && subgridValue_cmp != -INT_MAX))
		{
			cudaSolver->unusedCell_cuda[gid] = 0;
			atomicMax(&newSubgridValue, INT_MAX);
			atomicMax(&boundary[boundary_index], 1);
			cudaSolver->work_u_cuda[gid] = u_cmp;
		}
	}
	__syncthreads();

	if(threadIdx.x+threadIdx.y == 0)
	{
		if(subgridValue == INT_MAX && newSubgridValue !=0)
			cudaSolver->setSubgridValueCUDA(blockIdx.y*gridDim.x + blockIdx.x, -INT_MAX);

		cudaSolver->setBoundaryConditionCUDA(blockIdx.y*gridDim.x + blockIdx.x, FROM_NORTH * boundary[0] +
																				FROM_EAST * boundary[1] +
																				FROM_WEST * boundary[2] +
																				FROM_SOUTH * boundary[3]);
	}
}



template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void synchronize2CUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver)
{
	if(blockIdx.x+blockIdx.y ==0)
	{
		cudaSolver->currentStep = cudaSolver->currentStep + 1;
		*(cudaSolver->runcuda) = 0;
	}

	int stepValue = cudaSolver->currentStep + 4;
	if( cudaSolver->getSubgridValueCUDA(blockIdx.y*gridDim.x + blockIdx.x) == -INT_MAX )
			cudaSolver->setSubgridValueCUDA(blockIdx.y*gridDim.x + blockIdx.x, stepValue);

	atomicMax((cudaSolver->runcuda),cudaSolver->getBoundaryConditionCUDA(blockIdx.y*gridDim.x + blockIdx.x));
}








template< typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void initCUDA( tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* cudaSolver, double* ptr , int* ptr2, int* ptr3)
{
	cudaSolver->work_u_cuda = ptr;
	cudaSolver->unusedCell_cuda = ptr3;
	cudaSolver->subgridValues_cuda =(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*sizeof(int));
	cudaSolver->boundaryConditions_cuda =(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*sizeof(int));
	cudaSolver->runcuda = ptr2;
	*(cudaSolver->runcuda) = 1;
	cudaSolver->currentStep = 1;
	printf("GPU memory allocated.\n");

	for(int i = 0; i < cudaSolver->gridCols*cudaSolver->gridRows; i++)
	{
		cudaSolver->subgridValues_cuda[i] = INT_MAX;
		cudaSolver->boundaryConditions_cuda[i] = 0;
	}

	printf("GPU memory initialized.\n");
}




template< typename SchemeHost, typename SchemeDevice, typename Device >
__global__
void initRunCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller)

{


	extern __shared__ double u[];

	int i = blockIdx.y * gridDim.x + blockIdx.x;
	int l = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int containsCurve;
	if(l == 0)
		containsCurve = 0;

	caller->getSubgridCUDA(i,caller, &u[l]);
	__syncthreads();
	if(u[0] * u[l] <= 0.0)
	{
		atomicMax( &containsCurve, 1);
	}

	__syncthreads();
	if(containsCurve == 1)
	{
		caller->runSubgridCUDA(0,u,i);
		__syncthreads();
		caller->insertSubgridCUDA(u[l],i);
		__syncthreads();
		if(l == 0)
			caller->setSubgridValueCUDA(i, 4);
	}


}





template< typename SchemeHost, typename SchemeDevice, typename Device >
__global__
void runCUDA(tnlParallelEikonalSolver<SchemeHost, SchemeDevice, Device, double, int >* caller)
{
	extern __shared__ double u[];
	int i = blockIdx.y * gridDim.x + blockIdx.x;
	int l = threadIdx.y * blockDim.x + threadIdx.x;

	if(caller->getSubgridValueCUDA(i) != INT_MAX && caller->getSubgridValueCUDA(i) >= 0)
	{
		caller->getSubgridCUDA(i,caller, &u[l]);
		int bound = caller->getBoundaryConditionCUDA(i);
		if(caller->getSubgridValueCUDA(i) == caller->currentStep+4)
		{
			if(bound & FROM_NORTH)
			{
				caller->runSubgridCUDA(FROM_NORTH,u,i);
				__syncthreads();
				caller->updateSubgridCUDA(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & FROM_EAST )
			{
				caller->runSubgridCUDA(FROM_EAST,u,i);
				__syncthreads();
				caller->updateSubgridCUDA(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & FROM_WEST)
			{
				caller->runSubgridCUDA(FROM_WEST,u,i);
				__syncthreads();
				caller->updateSubgridCUDA(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & FROM_SOUTH)
			{
				caller->runSubgridCUDA(FROM_SOUTH,u,i);
				__syncthreads();
				caller->updateSubgridCUDA(i,caller, &u[l]);
				__syncthreads();
			}
		}



		if( ((bound & FROM_EAST) || (bound & FROM_NORTH) ))
		{
			caller->runSubgridCUDA(FROM_NORTH + FROM_EAST,u,i);
			__syncthreads();
			caller->updateSubgridCUDA(i,caller, &u[l]);
			__syncthreads();
		}
		if( ((bound & FROM_WEST) || (bound & FROM_NORTH) ))
		{
			caller->runSubgridCUDA(FROM_NORTH + FROM_WEST,u,i);
			__syncthreads();
			caller->updateSubgridCUDA(i,caller, &u[l]);
			__syncthreads();
		}
		if( ((bound & FROM_EAST) || (bound & FROM_SOUTH) ))
		{
			caller->runSubgridCUDA(FROM_SOUTH + FROM_EAST,u,i);
			__syncthreads();
			caller->updateSubgridCUDA(i,caller, &u[l]);
			__syncthreads();
		}
		if(   (bound & FROM_WEST) || (bound & FROM_SOUTH) )
		{
			caller->runSubgridCUDA(FROM_SOUTH + FROM_WEST,u,i);
			__syncthreads();
			caller->updateSubgridCUDA(i,caller, &u[l]);
			__syncthreads();
		}

		if(l==0)
		{
			caller->setBoundaryConditionCUDA(i, 0);
			caller->setSubgridValueCUDA(i, caller->getSubgridValueCUDA(i) - 1 );
		}

	}

}

#endif /*HAVE_CUDA*/

#endif /* TNLPARALLELEIKONALSOLVER_IMPL_H_ */
