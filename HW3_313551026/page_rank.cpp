#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {
  /*
    For PP students: Implement the page rank algorithm here.  You
    are expected to parallelize the algorithm using openMP.  Your
    solution may need to allocate (and free) temporary arrays.

    Basic page rank pseudocode is provided below to get you started:

    // initialization: see example code above
    score_old[vi] = 1/numNodes;

    while (!converged) {

      // compute score_new[vi] for all nodes vi:
      score_new[vi] = sum over all nodes vj reachable from incoming edges
                        { score_old[vj] / number of edges leaving vj  }
      score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

      score_new[vi] += sum over all nodes v in graph with no outgoing edges
                        { damping * score_old[v] / numNodes }

      // compute how much per-node scores have changed
      // quit once algorithm has converged

      global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
      converged = (global_diff < convergence)
    }

  */

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i) {
      solution[i] = equal_prob;
  }

  int* no_outgoing_nodes = (int*)malloc(numNodes * sizeof(int));
  int cnt = 0;
  for(int v = 0; v < numNodes; ++v){
    if(outgoing_size(g, v) == 0){
      no_outgoing_nodes[cnt++] = v;
    }
  }

  double global_diff, no_outgoing_sum;
  double *solution_new = (double*)malloc(numNodes * sizeof(double));

  do{
    global_diff = 0.0;
    no_outgoing_sum = 0.0;

    #pragma omp parallel for
    for(int vi = 0; vi < numNodes; ++vi){
      const Vertex *start = incoming_begin(g, vi);
      const Vertex *end = incoming_end(g, vi);
      double new_score = 0.0;
      for(const Vertex *incoming = start; incoming != end; incoming++){
        new_score += solution[*incoming] / outgoing_size(g, *incoming);
      }
      solution_new[vi] = (damping * new_score) + (1.0 - damping) / numNodes;
    }

    #pragma omp parallel for reduction(+:no_outgoing_sum)
    for(int i = 0; i < cnt; ++i){
      no_outgoing_sum += damping * solution[no_outgoing_nodes[i]] / numNodes;
    }

    #pragma omp parallel for
    for(int vi = 0; vi < numNodes; ++vi){
      solution_new[vi] += no_outgoing_sum;
    }

    #pragma omp parallel for reduction(+:global_diff)
    for(int vi = 0; vi < numNodes; ++vi){
      global_diff += fabs(solution_new[vi] - solution[vi]);
    }

    #pragma omp parallel for
    for(int vi = 0; vi < numNodes; ++vi){
      solution[vi] = solution_new[vi];
    }
    
  }while(global_diff >= convergence);

  free(no_outgoing_nodes);
  free(solution_new);
}