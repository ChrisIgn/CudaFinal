/*
 * Programa para calcular la pseudo-inversa de una matriz usando paralelización CUDA
 * Autores: Christian Delgado, Rafael Morales
 * Fecha: 09-07-2025
 * Ramo: Computación de Alto Rendimiento
 * Docente: Sergio Antonio Baltierra Valenzuela
 *
 * Compilar:
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define BLOCK 16;
//#define GRID  16;

//Funcion para escribir la matriz resultante en el archivo de salida
__host__  void printMatrix(const char *label, int m, int n, double *matriz){
    FILE *archivo = fopen("salida.sal", "w");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return;
    }
    fprintf(archivo, "%s\n", label);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(archivo, "%lf ", matriz[i * n + j]);
        }
        fprintf(archivo, "\n");
    }
    fclose(archivo);

}

/*************************************************************************************/


// Inversa de la Matriz
#define IDX(i, j, width) ((i) * (width) + (j))

__global__ void normalizar_fila(double* aug, int m, int fila, int ancho) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < ancho) {
        double pivote = aug[IDX(fila, fila, ancho)];
        aug[IDX(fila, j, ancho)] /= pivote;
    }
}

__global__ void eliminar_filas_kernel_2d(double* aug, int m, int fila_pivote, int ancho) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // filas
    int j = blockIdx.x * blockDim.x + threadIdx.x; // columnas

    if (i < m && i != fila_pivote && j < ancho) {
        double factor = aug[IDX(i, fila_pivote, ancho)];
        double valor_pivote = aug[IDX(fila_pivote, j, ancho)];
        aug[IDX(i, j, ancho)] -= factor * valor_pivote;
    }
}

__host__ int calcular_inversa_cuda(int m, double* matriz_h, double* matriz_inv_h) {
    const double EPS = 1e-16;
    int ancho = 2 * m;

    size_t size_aug = sizeof(double) * m * ancho;
    double* aug_h = (double*)malloc(size_aug);
    if (!aug_h) {
        fprintf(stderr, "No se pudo asignar memoria en host\n");
        return 1;
    }

    // Construir matriz aumentada [A | I]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            aug_h[IDX(i, j, ancho)] = matriz_h[IDX(i, j, m)];
            aug_h[IDX(i, j + m, ancho)] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Copiar matriz aumentada al device
    double* aug_d;
    cudaMalloc(&aug_d, size_aug);
    cudaMemcpy(aug_d, aug_h, size_aug, cudaMemcpyHostToDevice);

    for (int col = 0; col < m; col++) {
        // Copiar parte de la matriz para búsqueda de pivote
        cudaMemcpy(aug_h, aug_d, size_aug, cudaMemcpyDeviceToHost);

        int piv = col;
        for (int k = col + 1; k < m; k++) {
            if (fabs(aug_h[IDX(k, col, ancho)]) > fabs(aug_h[IDX(piv, col, ancho)])) {
                piv = k;
            }
        }

        if (fabs(aug_h[IDX(piv, col, ancho)]) < EPS) {
            fprintf(stderr, "Matriz no invertible (pivote cero)\n");
            free(aug_h);
            cudaFree(aug_d);
            return 1;
        }

        // Intercambiar filas si es necesario
        if (piv != col) {
            for (int j = 0; j < ancho; j++) {
                double tmp = aug_h[IDX(col, j, ancho)];
                aug_h[IDX(col, j, ancho)] = aug_h[IDX(piv, j, ancho)];
                aug_h[IDX(piv, j, ancho)] = tmp;
            }
        }

        // Copiar cambios de nuevo al device
        cudaMemcpy(aug_d, aug_h, size_aug, cudaMemcpyHostToDevice);

        // Normalizar fila pivote
        int blockSize = 256;
        int gridSize = (ancho + blockSize - 1) / blockSize;
        normalizar_fila<<<gridSize, blockSize>>>(aug_d, m, col, ancho);
        cudaDeviceSynchronize();

        // Eliminar otras filas
        eliminar_filas_kernel_2d<<<(m + 255) / 256, 256>>>(aug_d, m, col, ancho);
        cudaDeviceSynchronize();
    }

    // Copiar la matriz resultante al host
    cudaMemcpy(aug_h, aug_d, size_aug, cudaMemcpyDeviceToHost);

    // Extraer inversa
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            matriz_inv_h[IDX(i, j, m)] = aug_h[IDX(i, j + m, ancho)];
        }
    }

    // Liberar memoria
    free(aug_h);
    cudaFree(aug_d);

    return 0;
}

/*************************************************************************************/

// 
__global__ void eliminar_filas(double* A, int m, int n, int rank, int col_pivote, double* fila_pivote) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && i != rank) {
        double factor = A[i * n + col_pivote] / fila_pivote[col_pivote];
        for (int j = 0; j < n; j++) {
            A[i * n + j] -= factor * fila_pivote[j];
        }
    }
}

// Funcion para el calculo del Rango
int calcular_rango_secuencial(int m, int n, double* A_original) {
    const double EPS = 1e-16;

    // Crear copia local de la matriz para no modificar la original
    double* A = (double*)malloc(m * n * sizeof(double));
    if (!A) {
        fprintf(stderr, "Error: no se pudo reservar memoria para la matriz.\n");
        return -1;
    }
    memcpy(A, A_original, m * n * sizeof(double));

    int rank = 0;

    for (int i = 0; i < n; i++) {
        int pivote = -1;

        // Buscar fila con pivote válido en la columna actual
        for (int j = rank; j < m; j++) {
            if (fabs(A[j * n + i]) > EPS) {
                pivote = j;
                break;
            }
        }

        if (pivote != -1) {
            // Intercambiar filas si es necesario
            if (pivote != rank) {
                for (int k = 0; k < n; k++) {
                    double tmp = A[rank * n + k];
                    A[rank * n + k] = A[pivote * n + k];
                    A[pivote * n + k] = tmp;
                }
            }

            // Eliminar elementos de la columna actual en otras filas
            for (int j = 0; j < m; j++) {
                if (j != rank) {
                    double factor = A[j * n + i] / A[rank * n + i];
                    for (int k = 0; k < n; k++) {
                        A[j * n + k] -= factor * A[rank * n + k];
                    }
                }
            }

            // Aumentar el rango
            rank++;
        }
    }

    free(A);
    return rank;
}

// Calcular Transpuesta de matriz
__global__ void transpuesta_cuda(int m, int n, double *matriz_mn_d, double *matriz_nm_d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        matriz_nm_d[j * m + i] = matriz_mn_d[i * n + j];
    }
}

// Calcular la Multiplicación de matrices
#define TILE_WIDTH 32

__global__ void matrizMul_shared(double *A, double *B, double *C, int m, int n, int p) {
    __shared__ double tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    double sum = 0.0;

    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < m && t * TILE_WIDTH + threadIdx.x < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < p && t * TILE_WIDTH + threadIdx.y < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * p + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

// ---------------------------------------------------------------
// Calcula la pseudoinversa de una matriz m×n utilizando CUDA.
// Soporta tanto caso de rango completo por filas (m) como por columnas (n).
// ---------------------------------------------------------------

__host__ void pseudoinversa_cuda(int m, int n, double *matriz_mn_h) {
    // ------------------------------
    // Reservar memoria
    // ------------------------------

    size_t size_mn = m * n * sizeof(double);
    size_t size_nm = n * m * sizeof(double);
    size_t size_mm = m * m * sizeof(double);
    size_t size_nn = n * n * sizeof(double);

    // Punteros en dispositivo
    double *matriz_mn_d, *matriz_nm_d, *matriz_cuadrada_d, *matriz_inv_d, *matriz_pseudo_inv_d;

    // Punteros en host
    double *matriz_nm_h, *matriz_cuadrada_h, *matriz_inv_h, *matriz_pseudo_inv_h;

    // Asignación en host
    matriz_nm_h = (double *)malloc(size_nm);
    matriz_pseudo_inv_h = (double *)malloc(size_nm);
    matriz_cuadrada_h = (double *)malloc((m < n ? size_mm : size_nn));
    matriz_inv_h = (double *)malloc((m < n ? size_mm : size_nn));

    // Asignación en dispositivo
    cudaMalloc(&matriz_mn_d, size_mn);
    cudaMalloc(&matriz_nm_d, size_nm);
    cudaMalloc(&matriz_pseudo_inv_d, size_nm);
    cudaMalloc(&matriz_inv_d, (m < n ? size_mm : size_nn));
    cudaMalloc(&matriz_cuadrada_d, (m < n ? size_mm : size_nn));

    // Copiar matriz original al dispositivo
    cudaMemcpy(matriz_mn_d, matriz_mn_h, size_mn, cudaMemcpyHostToDevice);

    // ------------------------------
    // Configuración de bloques e hilos
    // ------------------------------
    dim3 blockDim(32, 32);
    dim3 gridDimTrans((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // ------------------------------
    // Calcular transpuesta: Aᵗ (n×m)
    // ------------------------------
    transpuesta_cuda<<<gridDimTrans, blockDim>>>(m, n, matriz_mn_d, matriz_nm_d);
    cudaMemcpy(matriz_nm_h, matriz_nm_d, size_nm, cudaMemcpyDeviceToHost);

    // ------------------------------
    // Calcular el rango de la matriz original
    // ------------------------------
    int rango = calcular_rango_secuencial(m, n, matriz_mn_h);

    // ------------------------------
    // Caso 1: Rango completo por filas (rango == m)
    // A⁺ = Aᵗ * (A * Aᵗ)⁻¹
    // ------------------------------
    if (rango == m) {

        // Calcular A * Aᵗ → matriz_cuadrada_d (m×m)
        dim3 gridDimMul((m + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
        matrizMul_shared<<<gridDimMul, blockDim>>>(matriz_mn_d, matriz_nm_d, matriz_cuadrada_d, m, n, m);
        cudaDeviceSynchronize();
        cudaMemcpy(matriz_cuadrada_h, matriz_cuadrada_d, size_mm, cudaMemcpyDeviceToHost);

        // Calcular inversa de A * Aᵗ
        int respuesta = calcular_inversa_cuda(m, matriz_cuadrada_h, matriz_inv_h);
        cudaMemcpy(matriz_inv_d, matriz_inv_h, size_mm, cudaMemcpyHostToDevice);

        // Calcular Aᵗ * (A * Aᵗ)⁻¹ → Pseudoinversa
        dim3 gridDimPI((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
        matrizMul_shared<<<gridDimPI, blockDim>>>(matriz_nm_d, matriz_inv_d, matriz_pseudo_inv_d, n, m, m);
        cudaDeviceSynchronize();

        // Copiar resultado al host
        cudaMemcpy(matriz_pseudo_inv_h, matriz_pseudo_inv_d, size_nm, cudaMemcpyDeviceToHost);
        printMatrix("R", n, m, matriz_pseudo_inv_h);

        free(matriz_inv_h);
    }

    // ------------------------------
    // Caso 2: Rango completo por columnas (rango == n)
    // A⁺ = (Aᵗ * A)⁻¹ * Aᵗ
    // ------------------------------
    if (rango == n) {

        // Calcular Aᵗ * A → matriz_cuadrada_d (n×n)
        dim3 gridDimMul((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
        matrizMul_shared<<<gridDimMul, blockDim>>>(matriz_nm_d, matriz_mn_d, matriz_cuadrada_d, n, m, n);
        cudaDeviceSynchronize();
        cudaMemcpy(matriz_cuadrada_h, matriz_cuadrada_d, size_nn, cudaMemcpyDeviceToHost);

        // Calcular inversa de Aᵗ * A
        int respuesta = calcular_inversa_cuda(n, matriz_cuadrada_h, matriz_inv_h);
        cudaMemcpy(matriz_inv_d, matriz_inv_h, size_nn, cudaMemcpyHostToDevice);

        // Calcular (Aᵗ * A)⁻¹ * Aᵗ → Pseudoinversa
        dim3 gridDimPI((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
        matrizMul_shared<<<gridDimPI, blockDim>>>(matriz_inv_d, matriz_nm_d, matriz_pseudo_inv_d, n, n, m);
        cudaDeviceSynchronize();

        cudaMemcpy(matriz_pseudo_inv_h, matriz_pseudo_inv_d, size_nm, cudaMemcpyDeviceToHost);
        printMatrix("L", n, m, matriz_pseudo_inv_h);

        free(matriz_inv_h);
    }

    // ------------------------------
    // Liberación de memoria (device y host)
    // ------------------------------
    cudaFree(matriz_mn_d);
    cudaFree(matriz_nm_d);
    cudaFree(matriz_cuadrada_d);
    cudaFree(matriz_inv_d);
    cudaFree(matriz_pseudo_inv_d);

    free(matriz_nm_h);
    free(matriz_cuadrada_h);
    free(matriz_pseudo_inv_h);
}

int main(int argc, char *argv[]) {
    int m, n;
    FILE *archivo = fopen("entrada_grande.ent", "r");

    if (!archivo)
    {
        perror("Error al abrir el archivo");
        return 1;
    }

    if (fscanf(archivo, "%d %d", &m, &n) != 2)
    {
        printf("Error al leer las dimenensiones del archivo\n");
        fclose(archivo);
        return 1;
    }

    // Reservar memoria en el host para la matriz original
    double *matriz_mn_h = (double *)malloc(m * n * sizeof(double));
    if (!matriz_mn_h) {
        printf("Error al reservar memoria para la matriz\n");
        fclose(archivo);
        return 1;
    }

    // Leer los elementos de la matriz desde el archivo
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(archivo, "%lf", &matriz_mn_h[i * n + j]) != 1) {
                printf("Error al leer el elemento (%d, %d)\n", i, j);
                fclose(archivo);
                free(matriz_mn_h);
                return 1;
            }
        }
    }
    fclose(archivo);

    // Medición del tiempo paralelo usando eventos CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    pseudoinversa_cuda(m, n, matriz_mn_h);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiempo_paralelo_ms;
    cudaEventElapsedTime(&tiempo_paralelo_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cálculo del speedup

    // Escribir resultados en tiempos.txt
    FILE *tiempos = fopen("tiempos.txt", "w");
    if (tiempos) {
        fprintf(tiempos, "T_paralelo: %.8f s\n", tiempo_paralelo_ms / 1000.0f);
        fclose(tiempos);
    } else {
        perror("Error al abrir tiempos.txt");
    }

    free(matriz_mn_h);
    return 0;
}