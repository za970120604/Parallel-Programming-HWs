#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances){
    // Assuming you have a way to initialize new_frontier
    // and that it has enough capacity allocated beforehand.

    // Maximum number of threads
    const int max_threads = omp_get_max_threads();
    int thread_local_counts[max_threads]; // Local counts for each thread
    int *thread_local_frontiers[max_threads]; // Local frontiers for each thread

    // Initialize local counts and frontiers for each thread
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_local_counts[thread_id] = 0; // Initialize count to zero
        thread_local_frontiers[thread_id] = (int*)malloc(sizeof(int) * g->num_nodes); // Allocate thread-local buffer
        int local_count = 0;

        #pragma omp for 
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // Attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                // Use compare_and_swap for thread-safe distance updates
                if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                {
                    // Add outgoing node to the local frontier
                    thread_local_frontiers[thread_id][local_count++] = outgoing; // Thread-local operation
                }
            }
        }

        // Save the local count for the thread
        thread_local_counts[thread_id] = local_count;
    }

    int totalCount = 0;
    int* count = (int*)malloc(sizeof(int) * max_threads);
    for (int t = 0; t < max_threads; ++t) {
      count[t] = totalCount;
      totalCount += thread_local_counts[t];
    }
    new_frontier->count = totalCount;

    // Parallel copy the data from `localList` to `frontier`
    #pragma omp parallel for
    for(int t = 0; t < max_threads; ++t){
        memcpy(new_frontier->vertices + count[t], thread_local_frontiers[t],
                thread_local_counts[t] * sizeof(int));
    }

    // Free the thread-local frontiers
    #pragma omp parallel for
    for(int t = 0; t < max_threads; t++)
    {
        free(thread_local_frontiers[t]); // Free each thread's local frontier
    }
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances){
    // Assuming you have a way to initialize new_frontier
    // and that it has enough capacity allocated beforehand.

    // Maximum number of threads
    const int max_threads = omp_get_max_threads();
    int thread_local_counts[max_threads]; // Local counts for each thread
    int *thread_local_frontiers[max_threads]; // Local frontiers for each thread
    int cur_distance = distances[frontier->vertices[0]];

    // Initialize local counts and frontiers for each thread
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_local_counts[thread_id] = 0; // Initialize count to zero
        thread_local_frontiers[thread_id] = (int*)malloc(sizeof(int) * g->num_nodes); // Allocate thread-local buffer
        int local_count = 0;

        #pragma omp for 
        for (int i = 0; i < g->num_nodes; i++){
            if(distances[i] == NOT_VISITED_MARKER){
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
                {
                    int incoming = g->incoming_edges[neighbor];
                    if (distances[incoming] == cur_distance){
                        distances[i] = distances[incoming] + 1;
                        thread_local_frontiers[thread_id][local_count++] = i;
                        break;
                    }
                }
            }
        }
        // Save the local count for the thread
        thread_local_counts[thread_id] = local_count;
    }

    int totalCount = 0;
    int* count = (int*)malloc(sizeof(int) * max_threads);
    for (int t = 0; t < max_threads; ++t) {
      count[t] = totalCount;
      totalCount += thread_local_counts[t];
    }
    new_frontier->count = totalCount;

    // Parallel copy the data from `localList` to `frontier`
    #pragma omp parallel for
    for(int t = 0; t < max_threads; ++t){
        memcpy(new_frontier->vertices + count[t], thread_local_frontiers[t],
                thread_local_counts[t] * sizeof(int));
    }

    // Free the thread-local frontiers
    #pragma omp parallel for
    for(int t = 0; t < max_threads; t++)
    {
        free(thread_local_frontiers[t]); // Free each thread's local frontier
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    
    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        // reset new_frontier
        vertex_set_clear(new_frontier);
        if(frontier->count < graph->num_nodes * 0.1)
        {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }
        else
        {
            bottom_up_step(graph, frontier, new_frontier, sol->distances);
        }

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
