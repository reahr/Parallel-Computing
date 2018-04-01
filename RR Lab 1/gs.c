#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */
float *xNew;
float *xOld;

/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input(char *);  /* Read input from file */
void get_range(int comm_sz, int * length, int * displs);
float get_data(float x[], float b, float row[], int my_i, int error[], int my_rank);
/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/*
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix() {
    int bigger = 0; /* Set to 1 if at least one diag element > sum  */
    int i, j;
    float sum = 0;
    float aii = 0;

    for (i = 0; i < num; i++) {
        sum = 0;
        aii = fabs(a[i][i]);

        for (j = 0; j < num; j++)
            if (j != i)
                sum += fabs(a[i][j]);

        if (aii < sum) {
            printf("The matrix will not converge.\n");
            exit(1);
        }

        if (aii > sum)
            bigger++;

    }

    if (!bigger) {
        printf("The matrix will not converge\n");
        exit(1);
    }
}

/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[]) {
    FILE *fp;
    int i, j;

    fp = fopen(filename, "r");
    if (!fp) {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }

    fscanf(fp, "%d ", &num);
    fscanf(fp, "%f ", &err);

    /* Now, time to allocate the matrices and vectors */
    a = (float **) malloc(num * sizeof(float *));
    if (!a) {
        printf("Cannot allocate a!\n");
        exit(1);
    }

    for (i = 0; i < num; i++) {
        a[i] = (float *) malloc(num * sizeof(float));
        if (!a[i]) {
            printf("Cannot allocate a[%d]!\n", i);
            exit(1);
        }
    }

    x = (float *) malloc(num * sizeof(float));
    xNew = (float *) malloc(num * sizeof(float));
    xOld = (float *) malloc(num * sizeof(float));

    if (!x) {
        printf("Cannot allocate x!\n");
        exit(1);
    }


    b = (float *) malloc(num * sizeof(float));
    if (!b) {
        printf("Cannot allocate b!\n");
        exit(1);
    }

    /* Now .. Filling the blanks */

    /* The initial values of Xs */
    for (i = 0; i < num; i++) {
        fscanf(fp, "%f ", &x[i]);
    }


    for (i = 0; i < num; i++) {
        for (j = 0; j < num; j++)
            fscanf(fp, "%f ", &a[i][j]);

        /* reading the b element */
        fscanf(fp, "%f ", &b[i]);
    }

    fclose(fp);

}

/******************************************************/
/**
 * This function will initialize two arrays that will contain the length and displacement of each process respectively
 * @param comm_sz total number of processes
 * @param length an array that will contain length per process
 * @param displs an array that will contain displacement per process (for allgatherv)
 */
void get_range(int comm_sz, int * length, int * displs){
    int numOfVals = num / comm_sz;
    int rem = num % comm_sz;

    //for loop, each process will need to do this because
    //need the displacement and length of previous rank for displacement calculations
    for (int i = 0; i < comm_sz; i++) {
        if (i == 0) {
            length[i] = numOfVals;
            displs[i] = 0; //since this is the first block of array
        } else {
            length[i] = numOfVals;
            //add on to end of last block so displacement is length of previous + their displacement
            displs[i] = length[i - 1] + displs[i - 1];
        }
        //each process will receive an extra element until rem=0 and the last processes gets num/comm_sz
        if (rem > 0) {
            length[i]++;
            rem--;
        }
    }
}

/**
 * This function calculates the new value of Xi using the global data given
 * @param x the global array x which has updated values of x PER iteration
 * @param b the given solution of the equation from array b
 * @param row the row that is used to calculate x, ie: X1 would use first row a[1]
 * @param my_i the Xi that we want to find
 * @param error array that contains if large error in any of the process' work
 * @param error my_rank process number which error[my_rank] will update if there is an error
 * @return the updated value of Xi
 */
float get_data(float x[], float b, float row[], int my_i, int error[], int my_rank) {
    float sum = 0;

    for (int i = 0; i < num; i++) {
        if (my_i == i) continue;
//            printf("Using %f and %f\n", row[i], x[i]);
        sum += row[i] * x[i];
//            printf("Summing up... %f\n", sum);
    }
//        printf("calculating: %f - %f / %f", b, sum, row[my_i]);

    float newX=(b - sum) / row[my_i];
    float newErr = fabsf((x[my_i] - newX) / newX);
    if (newErr > err) error[my_rank]=0; //set back to false

    return newX;
}

/************************************************************/

int main(int argc, char *argv[]) {
    int nit = 0; /* number of iterations */
    FILE *fp;
    char output[100] = "";
    int comm_sz;
    int my_rank;

    if (argc != 2) {
        printf("Usage: ./gs filename\n");
        exit(1);
    }

    /* Read the input file and fill the global data structure above */
    get_input(argv[1]);
    check_matrix();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    bool finished = false; //will be used in while loop
    int length[comm_sz]; //length of each process's array of X's
    int displs[comm_sz]; //for the sake of MPI_Allgatherv...
    int error[comm_sz];
    float newXs[num]; //contains all new X's

    //get range for each process and displacements for allgatherv
    get_range(comm_sz, length, displs);

//    printf("process %d will have length %d", my_rank, length[my_rank);

    //Each process's job per iteration:
    //1. Calculate the new x per amount of x's given to process
    //2. Concatenate each subarray into a new array of x's
    //3. Check if there is at least one error from any process, and if so we set back to false
    while (!finished) {
        nit++;
        finished = true; //reset every time so that we can check if all values in new X array is true
        int subArrayStart = displs[my_rank]; //start index of the new X array for each process
        error[my_rank]=1; //reset to true so that we can check if there's an error in calculation for new X's
        for (int i = 0; i < length[my_rank]; i++)
            //set the new X to calculated x, starting at displacement + i
            newXs[subArrayStart + i] = get_data(x, b[subArrayStart + i], a[subArrayStart + i], subArrayStart + i, error, my_rank);

        //allgather will collect subArrays in order of rank
        //every iteration will sync here and each process will now have the
        //new Xs they can check against -- hence no infinite loop/seg faults
        MPI_Allgatherv(&newXs[subArrayStart], length[my_rank], MPI_FLOAT, &newXs, length, displs, MPI_FLOAT,
                       MPI_COMM_WORLD);

        //gather a list of which contains bool per process
        MPI_Allgather(&error[my_rank], 1, MPI_INT, &error, 1, MPI_INT, MPI_COMM_WORLD);

        for (int i=0; i < comm_sz; i++){ //will be always iterate maximum comm size per process per iteration
            if (error[i]==0) {
                finished=false;
                break;
            }
        }

        //set x to new x's
        for (int i = 0; i < num; i++) {
            x[i] = newXs[i];
        }
    }

    /* Shut down MPI */
    MPI_Finalize();

    if (my_rank == 0) {
        /* Writing results to file */
        sprintf(output,"%d.sol",num);
        fp = fopen(output,"w");
        if(!fp)
        {
            printf("Cannot create the file %s\n", output);
            exit(1);
        }

        for(int i = 0; i < num; i++)
            fprintf(fp,"%f\n",x[i]);

        printf("total number of iterations: %d\n", nit);
        fclose(fp);
        printf("\n");
    }

    return 0;
}