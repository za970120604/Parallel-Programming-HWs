#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int finish(int *cur_cnt, int *old_cnt, int size) {
    int updated = 0;
    for (int i = 0; i < size; i++) {
        if (cur_cnt[i] != old_cnt[i]) {
            updated++;
        }
    }
    return (updated == size - 1);  // Exclude the master rank
}

int main(int argc, char **argv) {
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // MPI Initialization
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Calculate the number of tosses for each process
    long long toss_num = tosses / world_size;
    if(world_rank == 0){
        toss_num += tosses - world_size * toss_num;
    }

    int *in_circle;          // Shared memory for in-circle counts
    int *old_incircle;       // Local memory for tracking changes
    long long in_circle_num = 0; // Local in-circle count
    unsigned int seed = (world_rank) * time(NULL); // Seed for random number generator

    if (world_rank == 0) {
        // Master process
        MPI_Alloc_mem(world_size * sizeof(int), MPI_INFO_NULL, &in_circle);
        old_incircle = (int *)malloc(world_size * sizeof(int));
        for (int i = 0; i < world_size; i++) {
            in_circle[i] = 0;
            old_incircle[i] = 0;
        }
        MPI_Win_create(in_circle, world_size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        // Local tosses for master
        for (long long i = 0; i < toss_num; i++) {
            float x = rand_r(&seed) / ((float)RAND_MAX) * 2 - 1;
            float y = rand_r(&seed) / ((float)RAND_MAX) * 2 - 1;
            if (x * x + y * y <= 1) {
                in_circle_num++;
            }
        }

        // Wait for all processes to update their counts
        int ready = 0;
        while (!ready) {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = finish(in_circle, old_incircle, world_size);
            MPI_Win_unlock(0, win);
        }

        // Sum up the results from all processes
        for (int i = 0; i < world_size; i++) {
            in_circle_num += in_circle[i];
        }

        // Clean up
        free(old_incircle);
        MPI_Free_mem(in_circle);

    } else {
        // Worker processes
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        // Perform local tosses
        for (long long i = 0; i < toss_num; i++) {
            float x = rand_r(&seed) / ((float)RAND_MAX) * 2 - 1;
            float y = rand_r(&seed) / ((float)RAND_MAX) * 2 - 1;
            if (x * x + y * y <= 1) {
                in_circle_num++;
            }
        }

        // Update the master process with local results
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&in_circle_num, 1, MPI_INT, 0, world_rank, 1, MPI_INT, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0) {
        // Calculate PI
        pi_result = (double)(4.0 * in_circle_num) / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
