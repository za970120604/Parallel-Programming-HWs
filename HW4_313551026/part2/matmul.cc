#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstddef>

char reversed_int[10];
int world_rank = 0, world_size = 1;
int n_quotient, n_remainder;

void fast_write(int output){
    int i = 0;
    do{
        reversed_int[i++] = (output % 10) + '0';
    }while((output /= 10) > 0);

    while(--i >= 0){
        putchar_unlocked(reversed_int[i]);
    }
}

int get_row_start(const int &rank){
    return n_remainder + (rank - 1) * n_quotient;
}

void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0){
        in >> *n_ptr >> *m_ptr >> *l_ptr;

        MPI_Request req;
        for(int i = 1; i < world_size; i++){
            MPI_Isend(n_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(m_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &req);
            MPI_Isend(l_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &req);
        }

        n_quotient = (*n_ptr) / (world_size - 1);
        n_remainder = (*n_ptr) % (world_size - 1);

        *a_mat_ptr = (int *) aligned_alloc(32, (*n_ptr) * (*m_ptr) * sizeof(int));
        for(int row = 0; row < *n_ptr; row++){
            for(int column = 0; column < *m_ptr; column++){
                in >> (*a_mat_ptr)[row * (*m_ptr) + column];
            }
        }

        for(int i = 1; i < world_size; i++){
            MPI_Isend((*a_mat_ptr) + get_row_start(i) * (*m_ptr),
                      n_quotient * (*m_ptr),
                      MPI_INT,
                      i,
                      0,
                      MPI_COMM_WORLD,
                      &req);
        }

        *b_mat_ptr = (int *) aligned_alloc(32, (*m_ptr) * (*l_ptr) * sizeof(int));
        for(int row = 0; row < (*m_ptr); row++){
            for(int column = 0; column < (*l_ptr); column++){
                in >> (*b_mat_ptr)[column * (*m_ptr) + row];
            }
        }

        for(int i = 1; i < world_size; i++){
            MPI_Isend(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, i, 0, MPI_COMM_WORLD, &req);
        }
    } 
    else {
        MPI_Recv(n_ptr, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(m_ptr, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(l_ptr, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        n_quotient = (*n_ptr) / (world_size - 1);
        n_remainder = (*n_ptr) % (world_size - 1);

        *a_mat_ptr = (int *) aligned_alloc(32, (n_quotient) * (*m_ptr) * sizeof(int));
        *b_mat_ptr = (int *) aligned_alloc(32, (*m_ptr) * (*l_ptr) * sizeof(int));

        MPI_Recv(*a_mat_ptr, n_quotient * (*m_ptr), MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    a_mat = (int *) (__builtin_assume_aligned(a_mat, 32));
    b_mat = (int *) (__builtin_assume_aligned(b_mat, 32));

    if(world_rank == 0){
        int *c_mat = (int *) aligned_alloc(32, n * l * sizeof(int));
        c_mat = (int *) (__builtin_assume_aligned(c_mat, 32));

        for(int row = 0; row < n_remainder; row++){
            for(int column = 0; column < l; column++){
                int sum = 0;
                for(int i = 0; i < m; i++){
                    sum += a_mat[row * m + i] * b_mat[column * m + i];
                }
                c_mat[row * l + column] = sum;
            }
        }

        MPI_Request requests[world_size - 1];
        for(int source = 1; source < world_size; source++){
            MPI_Irecv(c_mat + get_row_start(source) * l,
                      n_quotient * l,
                      MPI_INT,
                      source,
                      MPI_ANY_TAG,
                      MPI_COMM_WORLD,
                      &requests[source - 1]);
        }

        MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);

        for(int i = 0; i < n * l; i++){
            fast_write(c_mat[i]);
            putchar_unlocked(' ');
            if((i + 1) % l == 0){
                putchar_unlocked('\n');
            }
        }
        free(c_mat);
    }
    else{
        int *c_mat = (int *) aligned_alloc(32, n_quotient * l * sizeof(int));
        c_mat = (int *) (__builtin_assume_aligned(c_mat, 32));

        for(int row = 0; row < n_quotient; row++){
            for(int column = 0; column < l; column++){
                int sum = 0;
                for(int i = 0; i < m; i++){
                    sum += a_mat[row * m + i] * b_mat[column * m + i];
                }
                c_mat[row * l + column] = sum;
            }
        }

        MPI_Send(c_mat, n_quotient * l, MPI_INT, 0, 0, MPI_COMM_WORLD);
        free(c_mat);
    }
}

void destruct_matrices(int *a_mat, int *b_mat) {
    free(a_mat);
    free(b_mat);
}
