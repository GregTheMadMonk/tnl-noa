/***************************************************************************
                          tnlParallelMapSolver2D_impl.h  -  description
                             -------------------
    begin                : Mar 22 , 2016
    copyright            : (C) 2016 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLPARALLELMAPSOLVER2D_IMPL_H_
#define TNLPARALLELMAPSOLVER2D_IMPL_H_


#include "tnlParallelMapSolver.h"
#include <core/mfilename.h>




#define MAP_SOLVER_MAX_VALUE 3



template< typename SchemeHost, typename SchemeDevice, typename Device>
tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::tnlParallelMapSolver()
{
	this->device = tnlHostDevice;  /////////////// tnlCuda Device --- vypocet na GPU, tnlHostDevice   ---    vypocet na CPU

#ifdef HAVE_CUDA
	if(this->device == tnlCudaDevice)
	{
	run_host = 1;
	}
#endif

}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::test()
{
/*
	for(int i =0; i < this->subgridValues.getSize(); i++ )
	{
		insertSubgrid(getSubgrid(i), i);
	}
*/
}

template< typename SchemeHost, typename SchemeDevice, typename Device>

bool tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::init( const Config::ParameterContainer& parameters )
{
	cout << "Initializating solver..." << endl;
	const String& meshLocation = parameters.getParameter <String>("mesh");
	this->mesh.load( meshLocation );

	this->n = parameters.getParameter <int>("subgrid-size");
	cout << "Setting N to " << this->n << endl;

	this->subMesh.setDimensions( this->n, this->n );
	this->subMesh.setDomain( Containers::StaticVector<2,double>(0.0, 0.0),
							 Containers::StaticVector<2,double>(mesh.template getSpaceStepsProducts< 1, 0 >()*(double)(this->n), mesh.template getSpaceStepsProducts< 0, 1 >()*(double)(this->n)) );

	this->subMesh.save("submesh.tnl");

	const String& initialCondition = parameters.getParameter <String>("initial-condition");
	this->u0.load( initialCondition );

	/* LOAD MAP */
	const String& mapFile = parameters.getParameter <String>("map");
	if(! this->map.load( mapFile ))
		cout << "Failed to load map file : " << mapFile << endl;


	this->delta = parameters.getParameter <double>("delta");
	this->delta *= mesh.template getSpaceStepsProducts< 1, 0 >()*mesh.template getSpaceStepsProducts< 0, 1 >();

	cout << "Setting delta to " << this->delta << endl;

	this->tau0 = parameters.getParameter <double>("initial-tau");
	cout << "Setting initial tau to " << this->tau0 << endl;
	this->stopTime = parameters.getParameter <double>("stop-time");

	this->cflCondition = parameters.getParameter <double>("cfl-condition");
	this -> cflCondition *= sqrt(mesh.template getSpaceStepsProducts< 1, 0 >()*mesh.template getSpaceStepsProducts< 0, 1 >());
	cout << "Setting CFL to " << this->cflCondition << endl;

	stretchGrid();
	this->stopTime /= (double)(this->gridCols);
	this->stopTime *= (1.0+1.0/((double)(this->n) - 2.0));
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
		cudaMalloc(&(this->cudaSolver), sizeof(tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >));
		cudaMemcpy(this->cudaSolver, this,sizeof(tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >), cudaMemcpyHostToDevice);

		double** tmpdev = NULL;
		cudaMalloc(&tmpdev, sizeof(double*));
		cudaMalloc(&(this->tmpw), this->work_u.getSize()*sizeof(double));
		cudaMalloc(&(this->tmp_map), this->map_stretched.getSize()*sizeof(double));
		cudaMalloc(&(this->runcuda), sizeof(int));
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;

		int* tmpUC;
		cudaMalloc(&(tmpUC), this->work_u.getSize()*sizeof(int));
		cudaMemcpy(tmpUC, this->unusedCell.getData(), this->unusedCell.getSize()*sizeof(int), cudaMemcpyHostToDevice);

		initCUDA2D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<1,1>>>(this->cudaSolver, (this->tmpw), (this->runcuda),tmpUC, tmp_map);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;

		double* tmpu = NULL;
		cudaMemcpy(&tmpu, tmpdev,sizeof(double*), cudaMemcpyDeviceToHost);
		cudaMemcpy((this->tmpw), this->work_u.getData(), this->work_u.getSize()*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy((this->tmp_map), this->map_stretched.getData(), this->map_stretched.getSize()*sizeof(double), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;

	}
#endif

	if(this->device == tnlHostDevice)
	{
		VectorType tmp_map;
		tmp_map.setSize(this->n * this->n);
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
				for( int j = 0; j < tmp_map.getSize(); j++)
				{
					tmp_map[j] = this->map_stretched[ (i / this->gridCols) * this->n*this->n*this->gridCols
										 + (i % this->gridCols) * this->n
										 + (j/this->n) * this->n*this->gridCols
										 + (j % this->n) ];
				}
				//cout << "Computing initial SDF on subgrid " << i << "." << endl;
				tmp[i] = runSubgrid(0, tmp[i],i,tmp_map);
				insertSubgrid(tmp[i], i);
				setSubgridValue(i, 4);
				//cout << "Computed initial SDF on subgrid " << i  << "." << endl;
			}
			containsCurve = false;

		}
	}
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;
		dim3 threadsPerBlock(this->n, this->n);
		dim3 numBlocks(this->gridCols,this->gridRows);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;
		initRunCUDA2D<SchemeTypeHost,SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock,3*this->n*this->n*sizeof(double)>>>(this->cudaSolver);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;

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

		synchronizeCUDA2D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock>>>(this->cudaSolver);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;
		synchronize2CUDA2D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,1>>>(this->cudaSolver);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;
	}

#endif
	cout << "Solver initialized." << endl;

	return true;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::run()
{
	if(this->device == tnlHostDevice)
	{
		while ((this->boundaryConditions.max() > 0 )/* || !end*/)
		{

#ifdef HAVE_OPENMP
#pragma omp parallel for num_threads(4) schedule(dynamic)
#endif
			for(int i = 0; i < this->subgridValues.getSize(); i++)
			{
				if(getSubgridValue(i) != INT_MAX)
				{
					VectorType tmp, tmp_map;
					tmp.setSize(this->n * this->n);
					tmp_map.setSize(this->n * this->n);
					for( int j = 0; j < tmp_map.getSize(); j++)
					{
						tmp_map[j] = this->map_stretched[ (i / this->gridCols) * this->n*this->n*this->gridCols
											 + (i % this->gridCols) * this->n
											 + (j/this->n) * this->n*this->gridCols
											 + (j % this->n) ];
					}

					if(getSubgridValue(i) == currentStep+4)
					{

						if(getBoundaryCondition(i) & 1)
						{
							tmp = getSubgrid(i);
							tmp = runSubgrid(1, tmp ,i,tmp_map);
							insertSubgrid( tmp, i);
							this->calculationsCount[i]++;
						}
						if(getBoundaryCondition(i) & 2)
						{
							tmp = getSubgrid(i);
							tmp = runSubgrid(2, tmp ,i,tmp_map);
							insertSubgrid( tmp, i);
							this->calculationsCount[i]++;
						}
						if(getBoundaryCondition(i) & 4)
						{
							tmp = getSubgrid(i);
							tmp = runSubgrid(4, tmp ,i,tmp_map);
							insertSubgrid( tmp, i);
							this->calculationsCount[i]++;
						}
						if(getBoundaryCondition(i) & 8)
						{
							tmp = getSubgrid(i);
							tmp = runSubgrid(8, tmp ,i,tmp_map);
							insertSubgrid( tmp, i);
							this->calculationsCount[i]++;
						}
					}
					else
					{

						if(getBoundaryCondition(i) == 1)
						{
							tmp = getSubgrid(i);
							tmp = runSubgrid(1, tmp ,i,tmp_map);
							insertSubgrid( tmp, i);
							this->calculationsCount[i]++;
						}
						if(getBoundaryCondition(i) == 2)
						{
							tmp = getSubgrid(i);
							tmp = runSubgrid(2, tmp ,i,tmp_map);
							insertSubgrid( tmp, i);
							this->calculationsCount[i]++;
						}
						if(getBoundaryCondition(i) == 4)
						{
							tmp = getSubgrid(i);
							tmp = runSubgrid(4, tmp ,i,tmp_map);
							insertSubgrid( tmp, i);
							this->calculationsCount[i]++;
						}
						if(getBoundaryCondition(i) == 8)
						{
							tmp = getSubgrid(i);
							tmp = runSubgrid(8, tmp ,i,tmp_map);
							insertSubgrid( tmp, i);
							this->calculationsCount[i]++;
						}
					}

					if(getBoundaryCondition(i) & 3)
					{
						//cout << "3 @ " << getBoundaryCondition(i) << endl;
						tmp = getSubgrid(i);
						tmp = runSubgrid(3, tmp ,i,tmp_map);
						insertSubgrid( tmp, i);
					}
					if(getBoundaryCondition(i) & 5)
					{
						//cout << "5 @ " << getBoundaryCondition(i) << endl;
						tmp = getSubgrid(i);
						tmp = runSubgrid(5, tmp ,i,tmp_map);
						insertSubgrid( tmp, i);
					}
					if(getBoundaryCondition(i) & 10)
					{
						//cout << "10 @ " << getBoundaryCondition(i) << endl;
						tmp = getSubgrid(i);
						tmp = runSubgrid(10, tmp ,i,tmp_map);
						insertSubgrid( tmp, i);
					}
					if(getBoundaryCondition(i) & 12)
					{
						//cout << "12 @ " << getBoundaryCondition(i) << endl;
						tmp = getSubgrid(i);
						tmp = runSubgrid(12, tmp ,i,tmp_map);
						insertSubgrid( tmp, i);
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
		TNL_CHECK_CUDA_DEVICE;

		bool* tmpb;
		cudaMemcpy(&(this->run_host),this->runcuda,sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		TNL_CHECK_CUDA_DEVICE;

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
			TNL_CHECK_CUDA_DEVICE;
			start = std::clock();
			runCUDA2D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock,3*this->n*this->n*sizeof(double)>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			time_diff += (std::clock() - start) / (double)(CLOCKS_PER_SEC);

			//start = std::clock();
			synchronizeCUDA2D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			TNL_CHECK_CUDA_DEVICE;
			synchronize2CUDA2D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,1>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			TNL_CHECK_CUDA_DEVICE;
			//time_diff += (std::clock() - start) / (double)(CLOCKS_PER_SEC);

			cudaMemcpy(&run_host, (this->runcuda),sizeof(int), cudaMemcpyDeviceToHost);
		}
		cout << "Solving time was: " << time_diff << endl;

		cudaMemcpy(this->work_u.getData()/* test*/, (this->tmpw), this->work_u.getSize()*sizeof(double), cudaMemcpyDeviceToHost);

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
		cudaFree(this->tmp_map);
		cudaFree(this->cudaSolver);
	}
#endif

}

//north - 1, east - 2, west - 4, south - 8
template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::synchronize() //needs fix ---- maybe not anymore --- but frankly: yeah, it does -- aaaa-and maybe fixed now
{
	cout << "Synchronizig..." << endl;
	int tmp1, tmp2;
	int grid1, grid2;

//	if(this->currentStep & 1)
//	{
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

//	}
//	else
//	{
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
//	}


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
int tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::getOwner(int i) const
{

	return (i / (this->gridCols*this->n*this->n))*this->gridCols + (i % (this->gridCols*this->n))/this->n;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
int tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::getSubgridValue( int i ) const
{
	return this->subgridValues[i];
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::setSubgridValue(int i, int value)
{
	this->subgridValues[i] = value;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
int tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::getBoundaryCondition( int i ) const
{
	return this->boundaryConditions[i];
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::setBoundaryCondition(int i, int value)
{
	this->boundaryConditions[i] = value;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::stretchGrid()
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
	if(!this->map_stretched.setSize(stretchedSize))
		cerr << "Could not allocate memory for stretched map." << endl;
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


		if(fabs(this->u0[i-k]) < mesh.template getSpaceStepsProducts< 1, 0 >()+mesh.template getSpaceStepsProducts< 0, 1 >() )
			this->work_u[i] = this->u0[i-k];
		else
			this->work_u[i] = sign(this->u0[i-k])*MAP_SOLVER_MAX_VALUE;

		this->map_stretched[i] = this->map[i-k];
	}


	cout << "Grid stretched." << endl;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::contractGrid()
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
typename tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::VectorType
tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::getSubgrid( const int i ) const
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
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::insertSubgrid( VectorType u, const int i )
{

	for( int j = 0; j < this->n*this->n; j++)
	{
		int index = (i / this->gridCols)*this->n*this->n*this->gridCols + (i % this->gridCols)*this->n + (j/this->n)*this->n*this->gridCols + (j % this->n);
		if( (fabs(this->work_u[index]) > fabs(u[j])) || (this->unusedCell[index] == 1) )
		{
			this->work_u[index] = u[j];
			this->unusedCell[index] = 0;
		}
	}
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
typename tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::VectorType
tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::runSubgrid( int boundaryCondition, VectorType u, int subGridID,VectorType map)
{

	VectorType fu;

	fu.setLike(u);
	fu.setValue( 0.0 );



	bool tmp = false;
	for(int i = 0; i < u.getSize(); i++)
	{
		if(u[0]*u[i] <= 0.0)
			tmp=true;
		int centerGID = (this->n*(subGridID / this->gridRows)+ (this->n >> 1))*(this->n*this->gridCols) + this->n*(subGridID % this->gridRows) + (this->n >> 1);
		if(this->unusedCell[centerGID] == 0 || boundaryCondition == 0)
			tmp = true;
	}


	double value = sign(u[0]) * u.absMax();

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



   double time = 0.0;
   double currentTau = this->tau0;
   double finalTime = this->stopTime;// + 3.0*(u.max() - u.min());
   if( time + currentTau > finalTime ) currentTau = finalTime - time;

   double maxResidue( 1.0 );
   tnlGridEntity<MeshType, 2, tnlGridEntityNoStencilStorage > Entity(subMesh);
   tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);

   for( int i = 0; i < u.getSize(); i ++ )
   {
		if(map[i] == 0.0)
		{
			u[i] = /*sign(u[l])**/MAP_SOLVER_MAX_VALUE;
		}
   }

   while( time < finalTime )
   {
      /****
       * Compute the RHS
       */

      for( int i = 0; i < fu.getSize(); i ++ )
      {
			Entity.setCoordinates(Containers::StaticVector<2,int>(i % subMesh.getDimensions().x(),i / subMesh.getDimensions().x()));
			Entity.refresh();
			neighborEntities.refresh(subMesh,Entity.getIndex());
			if(map[i] != 0.0)
				fu[ i ] = schemeHost.getValue( this->subMesh, i, Containers::StaticVector<2,int>(i % subMesh.getDimensions().x(),i / subMesh.getDimensions().x()), u, time, boundaryCondition,neighborEntities,map);
      }
      maxResidue = fu. absMax();


      if(maxResidue != 0.0)
    	  currentTau =  fabs(this -> cflCondition / maxResidue);


      if(currentTau > 1.0 * this->subMesh.template getSpaceStepsProducts< 1, 0 >())
      {
    	  currentTau = 1.0 * this->subMesh.template getSpaceStepsProducts< 1, 0 >();
      }


      if( time + currentTau > finalTime ) currentTau = finalTime - time;



      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  if(map[i] != 0.0)
    		  u[ i ] += currentTau * fu[ i ];
      }
      time += currentTau;

   }
   return u;
}


#ifdef HAVE_CUDA


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::getSubgridCUDA2D( const int i ,tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >* caller, double* a)
{
	int th = (blockIdx.y) * caller->n*caller->n*caller->gridCols
            + (blockIdx.x) * caller->n
            + threadIdx.y * caller->n*caller->gridCols
            + threadIdx.x;

	*a = caller->work_u_cuda[th];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::updateSubgridCUDA2D( const int i ,tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >* caller, double* a)
{
	int index = (blockIdx.y) * caller->n*caller->n*caller->gridCols
            + (blockIdx.x) * caller->n
            + threadIdx.y * caller->n*caller->gridCols
            + threadIdx.x;

	if( (fabs(caller->work_u_cuda[index]) > fabs(*a)) || (caller->unusedCell_cuda[index] == 1) )
	{
		caller->work_u_cuda[index] = *a;
		caller->unusedCell_cuda[index] = 0;

	}

	*a = caller->work_u_cuda[index];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::insertSubgridCUDA2D( double u, const int i )
{
		int index = (blockIdx.y)*this->n*this->n*this->gridCols
					+ (blockIdx.x)*this->n
					+ threadIdx.y*this->n*this->gridCols
					+ threadIdx.x;

		if( (fabs(this->work_u_cuda[index]) > fabs(u)) || (this->unusedCell_cuda[index] == 1) )
		{
			this->work_u_cuda[index] = u;
			this->unusedCell_cuda[index] = 0;

		}


}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::runSubgridCUDA2D( int boundaryCondition, double* u, int subGridID)
{

	__shared__ int tmp;
	__shared__ double value;
	volatile double* sharedTau = &u[blockDim.x*blockDim.y];
	double* map_local = &u[2*blockDim.x*blockDim.y];

	int i = threadIdx.x;
	int j = threadIdx.y;
	int l = threadIdx.y * blockDim.x + threadIdx.x;
	int gid = (blockDim.y*blockIdx.y + threadIdx.y)*blockDim.x*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;

	/* LOAD MAP */
	map_local[l]=this->map_stretched_cuda[gid];
	if(map_local[l] != 0.0)
		map_local[l] = 1.0/map_local[l];
	/* LOADED */

	bool computeFU = !((i == 0 && (boundaryCondition & 4)) or
			 (i == blockDim.x - 1 && (boundaryCondition & 2)) or
			 (j == 0 && (boundaryCondition & 8)) or
			 (j == blockDim.y - 1  && (boundaryCondition & 1)));

	if(l == 0)
	{
		tmp = 0;
		int centerGID = (blockDim.y*blockIdx.y + (blockDim.y>>1))*(blockDim.x*gridDim.x) + blockDim.x*blockIdx.x + (blockDim.x>>1);
		if(this->unusedCell_cuda[centerGID] == 0 || boundaryCondition == 0)
			tmp = 1;
	}
	__syncthreads();


	if(tmp !=1)
	{
		if(computeFU)
		{
			if(boundaryCondition == 4)
				u[l] = u[threadIdx.y * blockDim.x] ;//+ sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0 >()*(threadIdx.x);
			else if(boundaryCondition == 2)
				u[l] = u[threadIdx.y * blockDim.x + blockDim.x - 1] ;//+ sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0 >()*(this->n - 1 - threadIdx.x);
			else if(boundaryCondition == 8)
				u[l] = u[threadIdx.x] ;//+ sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0 >()*(threadIdx.y);
			else if(boundaryCondition == 1)
				u[l] = u[(blockDim.y - 1)* blockDim.x + threadIdx.x] ;//+ sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0 >()*(this->n - 1 - threadIdx.y);
		}
	}

   double time = 0.0;
   __shared__ double currentTau;
   double cfl = this->cflCondition;
   double fu = 0.0;

   double finalTime = this->stopTime;
   if(boundaryCondition == 0)
	   finalTime*=2.0;
   __syncthreads();

   tnlGridEntity<MeshType, 2, tnlGridEntityNoStencilStorage > Entity(subMesh);
   tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities(Entity);
   Entity.setCoordinates(Containers::StaticVector<2,int>(i,j));
   Entity.refresh();
   neighborEntities.refresh(subMesh,Entity.getIndex());


	if(map_local[l] == 0.0)
	{
		u[l] = /*sign(u[l])**/MAP_SOLVER_MAX_VALUE;
		computeFU = false;
	}
	__syncthreads();


   while( time < finalTime )
   {
	  sharedTau[l] = finalTime;

	  if(computeFU)
	  {
		  fu = schemeHost.getValueDev( this->subMesh, l, Containers::StaticVector<2,int>(i,j), u, time, boundaryCondition, neighborEntities, map_local);
	  	  sharedTau[l]=abs(cfl/fu);
	  }



      if(l == 0)
      {
    	  if(sharedTau[0] > 1.0 * this->subMesh.template getSpaceStepsProducts< 1, 0 >())	sharedTau[0] = 1.0 * this->subMesh.template getSpaceStepsProducts< 1, 0 >();
      }
      else if(l == blockDim.x*blockDim.y - 1)
    	  if( time + sharedTau[l] > finalTime )		sharedTau[l] = finalTime - time;


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
   }


}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::getOwnerCUDA2D(int i) const
{

	return ((i / (this->gridCols*this->n*this->n))*this->gridCols
			+ (i % (this->gridCols*this->n))/this->n);
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::getSubgridValueCUDA2D( int i ) const
{
	return this->subgridValues_cuda[i];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::setSubgridValueCUDA2D(int i, int value)
{
	this->subgridValues_cuda[i] = value;
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::getBoundaryConditionCUDA2D( int i ) const
{
	return this->boundaryConditions_cuda[i];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int>::setBoundaryConditionCUDA2D(int i, int value)
{
	this->boundaryConditions_cuda[i] = value;
}



//north - 1, east - 2, west - 4, south - 8

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void synchronizeCUDA2D(tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >* cudaSolver) //needs fix ---- maybe not anymore --- but frankly: yeah, it does -- aaaa-and maybe fixed now
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
		subgridValue = cudaSolver->getSubgridValueCUDA2D(blockIdx.y*gridDim.x + blockIdx.x);
		boundary[0] = 0;
		boundary[1] = 0;
		boundary[2] = 0;
		boundary[3] = 0;
		newSubgridValue = 0;
	}
	__syncthreads();



	if(		(threadIdx.x == 0 				 /*	&& !(cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.y == 0 				 /*	&& (cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.x == blockDim.x - 1 	 /*	&& !(cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.y == blockDim.y - 1 	 /*	&& (cudaSolver->currentStep & 1)*/) 		)
	{
		if(threadIdx.x == 0 && (blockIdx.x != 0)/* && !(cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid - 1];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA2D(blockIdx.y*gridDim.x + blockIdx.x - 1);
			boundary_index = 2;
		}

		if(threadIdx.x == blockDim.x - 1 && (blockIdx.x != gridDim.x - 1)/* && !(cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid + 1];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA2D(blockIdx.y*gridDim.x + blockIdx.x + 1);
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
		__threadfence();
		if(threadIdx.y == 0 && (blockIdx.y != 0)/* && (cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid - blockDim.x*gridDim.x];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA2D((blockIdx.y - 1)*gridDim.x + blockIdx.x);
			boundary_index = 3;
		}
		if(threadIdx.y == blockDim.y - 1 && (blockIdx.y != gridDim.y - 1)/* && (cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid + blockDim.x*gridDim.x];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA2D((blockIdx.y + 1)*gridDim.x + blockIdx.x);
			boundary_index = 0;
		}

		if((subgridValue == INT_MAX || fabs(u_cmp) + cudaSolver->delta < fabs(u) ) && (subgridValue_cmp != INT_MAX && subgridValue_cmp != -INT_MAX))
		{
			cudaSolver->unusedCell_cuda[gid] = 0;
			atomicMax(&newSubgridValue, INT_MAX);
			atomicMax(&boundary[boundary_index], 1);
			cudaSolver->work_u_cuda[gid] = u_cmp;
		}
	}
	__threadfence();
	__syncthreads();

	if(threadIdx.x+threadIdx.y == 0)
	{
		if(subgridValue == INT_MAX && newSubgridValue !=0)
			cudaSolver->setSubgridValueCUDA2D(blockIdx.y*gridDim.x + blockIdx.x, -INT_MAX);

		cudaSolver->setBoundaryConditionCUDA2D(blockIdx.y*gridDim.x + blockIdx.x, 	boundary[0] +
																				2 * boundary[1] +
																				4 * boundary[2] +
																				8 * boundary[3]);


		if(blockIdx.x+blockIdx.y ==0)
		{
			cudaSolver->currentStep += 1;
			*(cudaSolver->runcuda) = 0;
		}
	}

}



template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void synchronize2CUDA2D(tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >* cudaSolver)
{


	int stepValue = cudaSolver->currentStep + 4;
	if( cudaSolver->getSubgridValueCUDA2D(blockIdx.y*gridDim.x + blockIdx.x) == -INT_MAX )
			cudaSolver->setSubgridValueCUDA2D(blockIdx.y*gridDim.x + blockIdx.x, stepValue);

	atomicMax((cudaSolver->runcuda),cudaSolver->getBoundaryConditionCUDA2D(blockIdx.y*gridDim.x + blockIdx.x));
}








template< typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void initCUDA2D( tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >* cudaSolver, double* ptr , int* ptr2, int* ptr3, double* tmp_map_ptr)
{


	cudaSolver->work_u_cuda = ptr;
	cudaSolver->map_stretched_cuda = tmp_map_ptr;
	cudaSolver->unusedCell_cuda = ptr3;
	cudaSolver->subgridValues_cuda =(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*sizeof(int));
	cudaSolver->boundaryConditions_cuda =(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*sizeof(int));
	cudaSolver->runcuda = ptr2;
	*(cudaSolver->runcuda) = 1;

/* CHANGED !!!!!! from 1 to 0*/	cudaSolver->currentStep = 0;

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
void initRunCUDA2D(tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >* caller)

{
	extern __shared__ double u[];

	int i = blockIdx.y * gridDim.x + blockIdx.x;
	int l = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int containsCurve;
	if(l == 0)
		containsCurve = 0;


	caller->getSubgridCUDA2D(i,caller, &u[l]);
	__syncthreads();

	if(u[0] * u[l] <= 0.0)
		atomicMax( &containsCurve, 1);

	__syncthreads();
	if(containsCurve == 1)
	{
		caller->runSubgridCUDA2D(0,u,i);
		caller->insertSubgridCUDA2D(u[l],i);
		__syncthreads();
		if(l == 0)
			caller->setSubgridValueCUDA2D(i, 4);
	}


}





template< typename SchemeHost, typename SchemeDevice, typename Device >
__global__
void runCUDA2D(tnlParallelMapSolver<2,SchemeHost, SchemeDevice, Device, double, int >* caller)
{
	extern __shared__ double u[];
	int i = blockIdx.y * gridDim.x + blockIdx.x;
	int l = threadIdx.y * blockDim.x + threadIdx.x;
	int bound = caller->getBoundaryConditionCUDA2D(i);

	if(caller->getSubgridValueCUDA2D(i) != INT_MAX && bound != 0 && caller->getSubgridValueCUDA2D(i) > 0)
	{
		caller->getSubgridCUDA2D(i,caller, &u[l]);


		if(caller->getSubgridValueCUDA2D(i) == caller->currentStep+4)
		{
			if(bound & 1)
			{
				caller->runSubgridCUDA2D(1,u,i);
				caller->updateSubgridCUDA2D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & 2)
			{
				caller->runSubgridCUDA2D(2,u,i);
				caller->updateSubgridCUDA2D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & 4)
			{
				caller->runSubgridCUDA2D(4,u,i);
				caller->updateSubgridCUDA2D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & 8)
			{
				caller->runSubgridCUDA2D(8,u,i);
				caller->updateSubgridCUDA2D(i,caller, &u[l]);
				__syncthreads();
			}
		}
		else
		{

			if(bound == 1)
			{
				caller->runSubgridCUDA2D(1,u,i);
				caller->updateSubgridCUDA2D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound == 2)
			{
				caller->runSubgridCUDA2D(2,u,i);
				caller->updateSubgridCUDA2D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound == 4)
			{
				caller->runSubgridCUDA2D(4,u,i);
				caller->updateSubgridCUDA2D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound == 8)
			{
				caller->runSubgridCUDA2D(8,u,i);
				caller->updateSubgridCUDA2D(i,caller, &u[l]);
				__syncthreads();
			}
		}

		if(bound & 3)
		{
			caller->runSubgridCUDA2D(3,u,i);
			caller->updateSubgridCUDA2D(i,caller, &u[l]);
			__syncthreads();
		}
		if(bound & 5)
		{
			caller->runSubgridCUDA2D(5,u,i);
			caller->updateSubgridCUDA2D(i,caller, &u[l]);
			__syncthreads();
		}
		if(bound & 10)
		{
			caller->runSubgridCUDA2D(10,u,i);
			caller->updateSubgridCUDA2D(i,caller, &u[l]);
			__syncthreads();
		}
		if(bound & 12)
		{
			caller->runSubgridCUDA2D(12,u,i);
			caller->updateSubgridCUDA2D(i,caller, &u[l]);
			__syncthreads();
		}


		if(l==0)
		{
			caller->setBoundaryConditionCUDA2D(i, 0);
			caller->setSubgridValueCUDA2D(i, caller->getSubgridValueCUDA2D(i) - 1 );
		}


	}



}

#endif /*HAVE_CUDA*/

#endif /* TNLPARALLELMAPSOLVER2D_IMPL_H_ */
