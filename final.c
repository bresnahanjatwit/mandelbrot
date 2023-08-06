#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>
#include <omp.h>

extern void matToImage(char* filename, int* mat, int* dims);
extern void matToImageColor(char* filename, int* mat, int* dims);

int main(int argc, char **argv){
    
    int rank,numranks;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numranks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Status stat;

    int nx=56700;
    int ny=37800;
    int numColumns = 12;

    //master region
    if (rank == 0) {
	double bTime, eTime;
        int* matrix=(int*)malloc(nx*ny*sizeof(int));
        int* workerMatrix=(int*)malloc(nx*numColumns*sizeof(int));
        int start,end,nextStart,doneRanks,localStart;
        bool done=false;
        nextStart=0;
        int counter = 0;
        doneRanks=0;
	bTime=MPI_Wtime();
        for(int i=1;i<numranks;i++){
            start=nextStart;
            end=start+numColumns-1;
            nextStart+=numColumns;
            MPI_Send(&start,1,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Send(&end,1,MPI_INT,i,0,MPI_COMM_WORLD);
        }
        //wait to get values from every rank
        while(!done){
            MPI_Recv(workerMatrix,1+nx*numColumns,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&stat);
            localStart=workerMatrix[nx*numColumns];
            for(int i=0;i<numColumns;i++){
                for(int j=0;j<nx;j++){
		    if (i+localStart<ny)
                    matrix[(i+localStart)*nx+j]=workerMatrix[i*nx+j];
                }
            }
            
            //printf("Counter: %d | Start: %d | End: %d | lastUpdated: %d nextStart: %d\n",counter,start,end,localStart,nextStart);

            if(nextStart>=ny){ //no more work, set abort signal to ranks
                start=-1;
                doneRanks++; //counts the # of ranks that we've send
                             //the abort signal to
            }else{ //normal, compute start and end
                start=nextStart;
                end=start+numColumns-1;
                nextStart+=numColumns;
            }
            if(end>=ny){ //don't send too many!
                end=ny;
            }

            //sends
            MPI_Send(&start,1,MPI_INT,stat.MPI_SOURCE,0,MPI_COMM_WORLD);
	            MPI_Send(&end,1,MPI_INT,stat.MPI_SOURCE,0,MPI_COMM_WORLD);
            if(doneRanks==numranks-1){
                done=true;
            }
        }
	eTime=MPI_Wtime()-bTime;
	printf("Total Runtime with %d Nodes and %d Threads: %.6f\n",numranks,numColumns,eTime);
        int dims[2];
        dims[0]=ny;
        dims[1]=nx;
        matToImage("mandelbrot.jpg", matrix, dims);
        free(workerMatrix);
        free(matrix);
    }

    //worker region
    if (rank != 0) {
        int mystart, myend,localColumns;
        int* local_matrix=(int*)malloc(1+nx*numColumns*sizeof(int));
	double bTime,eTime;
        int maxIter=255;
        double xStart=-2;
        double xEnd=1;
        double yStart=-1;
        double yEnd=1;

	bTime=MPI_Wtime();
        while(true) { 
            MPI_Recv(&mystart,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat);
            MPI_Recv(&myend,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat);

            if (mystart == -1) break; //if start is invalid, we're done
            localColumns = myend-mystart+1;

            //Dynamically Allocate #of threads to be divisible by # of rows
            int numthreads = omp_get_num_threads();
            numthreads = localColumns;

            double x=0;
            double y=0;
            double x0=0;
            double y0=0;

            #pragma omp parallel num_threads(numthreads) firstprivate(x,y,x0,y0) shared(local_matrix)
            {
            int i = omp_get_thread_num();
            //for(int i=0;i<=(localColumns);i++){
                for(int j=0;j<nx;j++){
                    x0=xStart+(1.0*j/nx)*(xEnd-xStart);
                    y0=yStart+(1.0*(i+mystart)/ny)*(yEnd-yStart);
                    x=0;
                    y=0;
                    int iter=0;
                    while(iter<maxIter){
                        iter++;
                        double temp=x*x-y*y+x0;
                        y=2*x*y+y0;
                        x=temp;                
                        if(x*x+y*y>4){
                            break;
                        }
                    }
                    local_matrix[i*nx+j]=iter;
                }
            }
           //}
        local_matrix[nx*numColumns]=mystart;
        MPI_Send(local_matrix,1+numColumns*nx,MPI_INT,0,0,MPI_COMM_WORLD);
        }
    eTime=MPI_Wtime()-bTime;
    //printf("Runtime for Rank %d: %.6f\n",rank,eTime);
    free(local_matrix);
    }

    MPI_Finalize();
    return 0;
}

