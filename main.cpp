#include <iostream>
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

//N constant for size of arrays
const int N = 100000000;
float time1,time2  = 0;

// Function that calculates the dot product between two arrays
double DotProd(float* array1, float* array2, int size) {
    double result = 0.0f;
    for (int i = 0; i < size; ++i) {
        result += array1[i] * array2[i];
    }
    return result;
}
float DotProdInstrisics(float* array1, float* array2, int size){
    float result = 0.0f;
    //It is important to keep in mind that when using AVX", the registers have a size of 256 bits
    //which means you can only fit 8 32 bit floats
    int vectorLength = 8;
    __m256 sum_result = _mm256_setzero_ps(); // Initialize an AVX2 vector with zeros

    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_loadu_ps(&array1[i]); // Load 8 floats from a into a AVX2 vector
        __m256 b = _mm256_loadu_ps(&array2[i]); // Load 8 floats from b into a AVX2 vector
        __m256 product_result = _mm256_mul_ps(a, b); // Multiply the two vectors
        sum_result = _mm256_add_ps(sum_result, product_result); // Add the subproducts to find the complete one
    }
    // Add the elements of the resulting vector
    float sum_resultArray[8];
    _mm256_storeu_ps(sum_resultArray, sum_result); // Store the result of the vector in an array of floats
    for (int i = 0; i < vectorLength; i++) {
        result += sum_resultArray[i];
    }
    return result;
}
void No_Improvement(float* array1, float* array2){
    // Initialize arrays with random values
    double beginning_time, execution_time;
    // Measure execution time
    beginning_time = omp_get_wtime();
    // Calculate the dot product
    double result = DotProd(array1, array2, N);
    // Calculate runtime
    execution_time = omp_get_wtime() - beginning_time;
    time1 = execution_time;
    std::cout << "The Dot product without compilation optimization is equal to " <<std::setprecision(3)<< result << std::endl;
    std::cout << "The total Runtime without compilation optimization is equal to " << execution_time << " s" << std::endl;
    delete[] array1;
    delete[] array2;
}
void Apply_Intrinsics(float* array1, float* array2){
    // Initialize arrays with random values
    double beginning_time, execution_time;
    // Measure execution time
    beginning_time = omp_get_wtime();
    // Calculate the dot product
    float result = DotProdInstrisics(array1, array2, N);
    //Calculate runtime
    execution_time = omp_get_wtime() - beginning_time;
    time2 = execution_time;
    std::cout << "The Dot product when using intrinsics AVX2 is equal to " << result << std::endl;
    std::cout << "The total Runtime when using intrinsics AVX2 is equal to " << execution_time << " s" << std::endl;
    delete[] array1;
    delete[] array2;
}

int main() {
    if(fmod(N,4) == 0){
        //Define arrays of size N
        float* array1 = new float[N];
        float* array2 = new float[N];
        float* array3 = new float[N];
        float* array4 = new float[N];

        // Fill up with values
        for (int i = 0; i < N; ++i) {
            array1[i] = i;
            array2[i] = i;
            array3[i] = array1[i];
            array4[i] = array2[i];
        }
        // Execute the tests
        No_Improvement(array1, array2);
        Apply_Intrinsics(array3, array4);

        std::cout << "The time difference between the runtimes is equal to " << time1-time2  << " s" << std::endl;

    }else{
        std::cout << "The selected size is not a multiple of 4" << std::endl;
    }
    return 0;
}
