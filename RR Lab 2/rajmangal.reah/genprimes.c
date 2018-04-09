#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/**
 * Reah Rajmangal, rr2886
 * Lab 2, OpenMP
 */

int main(int argc, char **argv) {
    double tstart = 0.0, ttaken;
    FILE *fp;
    char output[100] = "";

    if (argc != 3) {
        printf("Usage: ./genprime N t\n");
        printf("N is a positive number bigger than 2 and less than or equal to 100,000\n");
        printf("t is positive number no more than 100\n");
        exit(1);
    }

    //read from command line
    int N = atoi(argv[1]);
    int t = atoi(argv[2]);

    int primes[N + 1]; //include last index in checking for prime

#pragma omp parallel for num_threads(t)
    //set all to true, rule out non-primes when beginning algorithm
    for (int i = 0; i <= N; i++) {
        primes[i] = 1;
    }

    //start algorithm, use same number of threads created from previous for-loop
    tstart = omp_get_wtime();
#pragma omp parallel for num_threads(t) default (none) shared (N, primes, t)
    //set to 0 as cross-out if it is not prime
    for (int i = 2; i <= N; i++) {
        if (primes[i]) { //only go into nested loop if initially marked as prime/remaining prime
            for (int j = 2; j <= (i + 1) / 2; j++) {
                //begin ruling out based on identity rule (2/2==1) and divisibility
                if ((i / j != 1) && (i % j == 0)) {
                    primes[i] = 0;
                }
            }
        }
    }
    //end algorithm

    ttaken = omp_get_wtime() - tstart;
    printf("Time take for the main part: %f\n", ttaken);

    //output to file
    sprintf(output, "%d.txt", N);
    fp = fopen(output, "w");
    if (!fp) {
        {
            printf("Cannot create the file %s\n", output);
            exit(1);
        }
    }

    int rank, previousPrime = 2; //first prime number checked is always 2
    for (int i = 2; i <= N; i++) {
        //primes[i] will be set to 1 if true so increase rank, set previous here
        if (primes[i]) {
            rank++;
            fprintf(fp, "%d, %d, %d\n", rank, i, i - previousPrime);
            previousPrime = i;
        }
    }

    fclose(fp);

    return 0;
}