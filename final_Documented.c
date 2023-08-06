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

    int nx=56700; //Max size to fill an array of 32 bits in size (2.147 billion)
    int ny=37800;
    int numRows = 12; //number of rows to be sent to each node (Should be less than 12 which is the max number of threads & divisible by ny)

    //master region
    if (rank == 0) {
        int* matrix=(int*)malloc(nx*ny*sizeof(int));
        int* workerMatrix=(int*)malloc(nx*numRows*sizeof(int));
        int start,end,nextStart,doneRanks,localStart;
        bool done=false;
        nextStart=0;
        doneRanks=0;

        for(int i=1;i<numranks;i++){ //initially volley of work
            start=nextStart;
            end=start+numRows-1;
            nextStart+=numRows;
            MPI_Send(&start,1,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Send(&end,1,MPI_INT,i,0,MPI_COMM_WORLD);
        }
        //wait to get values from every rank
        while(!done){
            MPI_Recv(workerMatrix,1+nx*numRows,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&stat);
            localStart=workerMatrix[nx*numRows];
            for(int i=0;i<numRows;i++){
                for(int j=0;j<nx;j++){
                    matrix[(i+localStart)*nx+j]=workerMatrix[i*nx+j];
                }
            }

            if(nextStart>=ny){ //no more work, set abort signal to ranks
                start=-1;
                doneRanks++; //counts the # of ranks that we've send
                             //the abort signal to
            }else{ //normal, compute start and end
                start=nextStart;
                end=start+numRows-1;
                nextStart+=numRows;
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

        int dims[2]; //end of master section
        dims[0]=ny;
        dims[1]=nx;
        matToImage("mandelbrot.jpg", matrix, dims); //create image
        free(workerMatrix); //free memory
        free(matrix);
    }

    //worker region
    if (rank != 0) {
        int mystart,myend,numthreads;
        int* local_matrix=(int*)malloc(1+nx*numRows*sizeof(int));
        int maxIter=255;
        double xStart=-2;
        double xEnd=1;
        double yStart=-1;
        double yEnd=1;


        while(true) { 
            MPI_Recv(&mystart,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat);
            MPI_Recv(&myend,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat);

            if (mystart == -1) break; //if start is invalid, we're done
            numthreads = myend-mystart+1; //Dynamically Allocate # of threads

            double x=0;
            double y=0;
            double x0=0;
            double y0=0;

            //OpenMP Section
            #pragma omp parallel num_threads(numthreads) firstprivate(x,y,x0,y0) shared(local_matrix)
            {
            int i = omp_get_thread_num(); //One row per thread

                for(int j=0;j<nx;j++){ //mandelbrot set calculations
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
        local_matrix[nx*numRows]=mystart; //Pass identifier of where to put this section of the image to the master
        MPI_Send(local_matrix,1+numRows*nx,MPI_INT,0,0,MPI_COMM_WORLD); 
        }
    free(local_matrix);
    } //End Worker Region
    //Finalize on all nodes
    MPI_Finalize();
    return 0;
}