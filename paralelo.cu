/*
 * Programa para calcular la pseudo-inversa de una matriz usando paralelización CUDA
 * Autores: Christian Delgado, Rafael Morales
 * Fecha: 09-07-2025
 * Ramo: Computación de Alto Rendimiento
 * Docente: Sergio Antonio Baltierra Valenzuela
 *
 * Compilar:
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int BLOCK = 16;
const int GRID = 16;

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

__global__ void eliminar_filas_kernel(double* aug, int m, int fila_pivote, int ancho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && i != fila_pivote) {
        double factor = aug[IDX(i, fila_pivote, ancho)];
        for (int j = 0; j < ancho; j++) {
            aug[IDX(i, j, ancho)] -= factor * aug[IDX(fila_pivote, j, ancho)];
        }
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
        eliminar_filas_kernel<<<(m + 255) / 256, 256>>>(aug_d, m, col, ancho);
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
__host__ int calcular_rango_cuda(int m, int n, double* A_original_h) {
    const double EPS = 1e-16;

    // Crear una copia de A_h para no modificar la original
    double* A_h = (double*)malloc(m * n * sizeof(double));
    memcpy(A_h, A_original_h, m * n * sizeof(double));

    // Copiar matriz al device
    double* A_d;
    cudaMalloc(&A_d, m * n * sizeof(double));
    cudaMemcpy(A_d, A_h, m * n * sizeof(double), cudaMemcpyHostToDevice);

    double* fila_pivote_h = (double*)malloc(n * sizeof(double));
    double* fila_pivote_d;
    cudaMalloc(&fila_pivote_d, n * sizeof(double));

    int rank = 0;

    for (int i = 0; i < n; i++) {
        // Buscar fila pivote desde el host
        int pivote = -1;
        cudaMemcpy(A_h, A_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

        for (int j = rank; j < m; j++) {
            if (fabs(A_h[j * n + i]) > EPS) {
                pivote = j;
                break;
            }
        }

        if (pivote != -1) {
            // Intercambiar filas pivote y actual
            if (pivote != rank) {
                for (int k = 0; k < n; k++) {
                    double tmp = A_h[rank * n + k];
                    A_h[rank * n + k] = A_h[pivote * n + k];
                    A_h[pivote * n + k] = tmp;
                }
                cudaMemcpy(A_d, A_h, m * n * sizeof(double), cudaMemcpyHostToDevice);
            }

            // Copiar fila pivote al host y luego al device
            cudaMemcpy(fila_pivote_h, &A_d[rank * n], n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(fila_pivote_d, fila_pivote_h, n * sizeof(double), cudaMemcpyHostToDevice);

            eliminar_filas<<<BLOCK, GRID>>>(A_d, m, n, rank, i, fila_pivote_d);
            cudaDeviceSynchronize();

            rank++;
        }
    }

    // Liberar memoria
    cudaFree(A_d);
    cudaFree(fila_pivote_d);
    free(fila_pivote_h);
    free(A_h);  // también libera la copia local

    return rank;
}

// 
__global__ void transpuesta_cuda(int m, int n, double *matriz_mn_d, double *matriz_nm_d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        matriz_nm_d[j * m + i] = matriz_mn_d[i * n + j];
    }
}

// multiplicación de matrices
__global__ void matrizMul_cuda(double *A, double *B, double *C, int m, int n, int p) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < p) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[k * p + j];
        }
        C[i * p + j] = sum;
    }
}

// 
__host__ void pseudoinversa_cuda(int m, int n, double *matriz_mn_h) {
    // Parametrización reserva de memoria
    size_t size_mn = m * n * sizeof(double);
    size_t size_nm = n * m * sizeof(double);
    size_t size_mm = m * m * sizeof(double);
    size_t size_nn = n * n * sizeof(double);

    double *matriz_mn_d, *matriz_nm_d, *matriz_cuadrada_d;
    double *matriz_nm_h, *matriz_cuadrada_h;

    // Reservar memoria en el host
    matriz_nm_h = (double *)malloc(size_nm);
    if (!matriz_nm_h) {
        fprintf(stderr, "Error al reservar memoria para matriz_nm_h\n");
        return;
    }
    matriz_cuadrada_h = (double *)malloc((m < n ? size_mm : size_nn));
    if (!matriz_cuadrada_h) {
        fprintf(stderr, "Error al reservar memoria para matriz_cuadrada_h\n");
        free(matriz_nm_h);
        return;
    }

    // Reservar memoria en el dispositivo
    cudaMalloc(&matriz_mn_d, size_mn);
    cudaMalloc(&matriz_nm_d, size_nm);
    cudaMalloc(&matriz_cuadrada_d, (m < n ? size_mm : size_nn));

    cudaMemcpy(matriz_mn_d, matriz_mn_h, size_mn, cudaMemcpyHostToDevice);

    // Configurar dimensiones de grid y block
    dim3 blockDim(BLOCK, GRID);
    dim3 gridDimTrans((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    dim3 gridDimMul((m < n ? m : n + blockDim.x - 1) / blockDim.x, (m < n ? m : n + blockDim.y - 1) / blockDim.y);

    // Calcular Transpuesta
    transpuesta_cuda<<<gridDimTrans, blockDim>>>(m, n, matriz_mn_d, matriz_nm_d);
    cudaMemcpy(matriz_nm_h, matriz_nm_d, size_nm, cudaMemcpyDeviceToHost);

    // Calcular Matriz inversa 
    int rango = calcular_rango_cuda(m, n, matriz_mn_h);
    if (rango == m) {
        // Calcular A^T A (m×m)
        matrizMul_cuda<<<gridDimMul, blockDim>>>(matriz_mn_d, matriz_nm_d, matriz_cuadrada_d, m, n, m);
        cudaDeviceSynchronize();
        cudaMemcpy(matriz_cuadrada_h, matriz_cuadrada_d, size_mm, cudaMemcpyDeviceToHost);
        // Imprimir matriz cuadrada (temporal, para depuración)
        printMatrix("A^T A", m, m, matriz_cuadrada_h);

        double *matriz_inv_h = (double *)malloc(n * n * sizeof(double));
        int respuesta = calcular_inversa_cuda(n, matriz_nm_h, matriz_inv_h);
        printMatrix("Matriz Transpuesta", n, n, matriz_inv_h);
        free(matriz_inv_h);
    }
    if (rango == n) {
        // Calcular A A^T (n×n)
        matrizMul_cuda<<<gridDimMul, blockDim>>>(matriz_nm_d, matriz_mn_d, matriz_cuadrada_d, n, m, n);
        cudaDeviceSynchronize();
        cudaMemcpy(matriz_cuadrada_h, matriz_cuadrada_d, size_nn, cudaMemcpyDeviceToHost);
        // Imprimir matriz cuadrada (temporal, para depuración)
        printMatrix("A A^T", n, n, matriz_cuadrada_h);

        double *matriz_inv_h = (double *)malloc(n * n * sizeof(double));
        int respuesta = calcular_inversa_cuda(n, matriz_nm_h, matriz_inv_h);
        printMatrix("Matriz Transpuesta", n, n, matriz_inv_h);
        free(matriz_inv_h);
    }

    // Liberacion de espacios de gpu
    cudaFree(matriz_mn_d);
    cudaFree(matriz_nm_d);
    cudaFree(matriz_cuadrada_d);

    // Liberación de espacios de host
    free(matriz_nm_h);
    free(matriz_cuadrada_h);

    return;
}

int main(int argc, char *argv[]) {
    int m, n;
    FILE *archivo = fopen("entrada.ent", "r");

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
    
    pseudoinversa_cuda(m, n, matriz_mn_h);

    return 0;
}