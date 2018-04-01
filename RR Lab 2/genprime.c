#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    double tstart=0.0, tend=0.0, ttaken;
    char filename[100];
    //read from command line
    int N= atoi(argv[1]);
    int t=atoi(argv[2]);
    int rank, interval, previousPrime=2;
    int primes[N+1]; //include last index in checking for prime
    printf("Num of threads: %d", omp_get_num_threads());
    tstart=opm_get_wtime();

#pragma omp parallel schedule(dynamic) num_threads(t)
    //we can have an array of N elements and set to 1 if it is a prime number
    for (int i=2; i <= N; i++){
#pragma omp parallel for num_threads(t)
        for (int j=2; j <= (i+1)/2; j++){
            if ((i/j !=0) && (i%j==0)){
                continue; //false, not a prime based on identity check and division
            }else{
                primes[i]=1;
            }
        }
    }

    ttaken=omp_get_wtime()-tstart;
    printf("Time take for the main part: %f\n", ttaken);
    sprintf(filename, "%d.txt", N);
    FILE *file=fopen(filename, "w");

    for (int i=2; i <=N; i++){
        //primes[i] will be set to 1 if true so increase rank, set previous here
        if (primes[i]){
            rank++;
            fprintf(file, "%d %d %d\n", rank, i, i-previousPrime);
            previousPrime=i;
        }
    }
    fclose(file);

    return 0;
}