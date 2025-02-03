#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <iomanip> 
#include <random>
#include <immintrin.h>

#ifndef __AVX2_AVAILABLE__
#define __AVX2_AVAILABLE__
#endif

#include "./include/SIMDInstructionSet.h"
#include "./include/Xoshiro256Plus.h"

typedef SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2> Xoshiro256PlusAVX2;
const __m256 rand_max = _mm256_set1_ps(RAND_MAX);
const __m256 one = _mm256_set1_ps(1.0f);

// Shared variable and mutex
long long inside_cnts = 0;
pthread_mutex_t mutex_lock;

double random_double_linear() {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(-1, 1);
    return distribution(generator);
}

void* SIMD_TOSS(void* SIMD_toss_cnts) {
    long long toss_num = *((long long*) SIMD_toss_cnts);
    long long local_SIMD_cnts = 0;

    Xoshiro256PlusAVX2 rng(std::rand());  // Initialize RNG once per thread
    alignas(32) float result[8];

    for (long long i = 0; i < toss_num; i++) {
        __m256 rand_float_x = _mm256_cvtepi32_ps(rng.next4().operator __m256i());
        __m256 rand_float_y = _mm256_cvtepi32_ps(rng.next4().operator __m256i());
        __m256 normalized_float_x = _mm256_div_ps(rand_float_x, rand_max); 
        __m256 normalized_float_y = _mm256_div_ps(rand_float_y, rand_max); 

        __m256 distance = _mm256_add_ps(_mm256_mul_ps(normalized_float_x, normalized_float_x), _mm256_mul_ps(normalized_float_y, normalized_float_y));  // x * x + y * y
        __m256 in_circle_mask = _mm256_cmp_ps(distance, one, _CMP_LE_OS);  // distance <= 1

        __m256 in_circle = _mm256_and_ps(one, in_circle_mask);
        __m256 in_circle_permute = _mm256_permute2f128_ps(in_circle, in_circle, 1);
        in_circle = _mm256_hadd_ps(in_circle, in_circle_permute);
        in_circle = _mm256_hadd_ps(in_circle, in_circle);
        in_circle = _mm256_hadd_ps(in_circle, in_circle);

        _mm256_store_ps(result, in_circle);

        local_SIMD_cnts += static_cast<long long>(result[0]);
    }

    pthread_mutex_lock(&mutex_lock);
    inside_cnts += local_SIMD_cnts;
    pthread_mutex_unlock(&mutex_lock);

    return nullptr;
}

int main(int argc, char* argv[]) {

    long long total_toss = strtol(argv[2], NULL, 10);
    long long number_of_threads = strtol(argv[1], NULL, 10);
    long long SIMD_toss_cnts_per_thread = (total_toss / 8) / number_of_threads;
    long long remain_toss_cnts = total_toss - SIMD_toss_cnts_per_thread * 8 * number_of_threads;

    // Initialize the mutex
    pthread_mutex_init(&mutex_lock, NULL);

    pthread_t threads[number_of_threads];
    long long* SIMD_cnts = (long long*)malloc(number_of_threads * sizeof(long long));

    // Create threads
    for (int i = 0; i < number_of_threads; i++) {
        SIMD_cnts[i] = SIMD_toss_cnts_per_thread;
        pthread_create(&(threads[i]), NULL, SIMD_TOSS, (void*)&(SIMD_cnts[i]));
    }

    // Join threads
    for (int i = 0; i < number_of_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(SIMD_cnts);

    // Handle remaining tosses
    for (int i = 0; i < remain_toss_cnts; i++) {
        double x = random_double_linear();
        double y = random_double_linear();
        if ((x * x + y * y) <= 1) {
            inside_cnts++;
        }
    }

    // Destroy the mutex after all threads have finished
    pthread_mutex_destroy(&mutex_lock);

    std::cout << std::fixed << std::setprecision(10);
    double pi_estimate = (static_cast<double>(inside_cnts) / total_toss) * 4.0;
    std::cout << pi_estimate << std::endl;

    return 0;
}
