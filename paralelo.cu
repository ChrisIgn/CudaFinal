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


__global__ void transpuesta_cuda(int m, int n, double *matriz_mn_d, double *matriz_nm_d) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        matriz_nm_d[j * m + i] = matriz_mn_d[i * n + j];
    }
}


int pseudoinversa_cuda(int m, int n, double *matriz_mn_h)
{
    // Parametrización reserva de memoria
    /* double *matriz_mn_d, *matriz_nm_d, *matriz_cuadrada_inv_d, *matriz_pseudo_inv_d; */
    double *matriz_mn_d, *matriz_nm_d;
    size_t size_mn = m * n * sizeof(double);
    size_t size_nm = n * m * sizeof(double);
    /* size_t size_mm = m * m * sizeof(double);
    size_t size_nn = n * n * sizeof(double); */

    // Reservar memoria en el dispositivo
    cudaMalloc(&matriz_mn_d, size_mn);
    cudaMalloc(&matriz_nm_d, size_nm);

    cudaMemcpy(matriz_mn_d, matriz_mn_h, size_mn, cudaMemcpyHostToDevice);

    // Configurar dimensiones de grid y block
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);


    // Calcular Transpuesta
    transpuesta_cuda<<<gridDim, blockDim>>>(m, n, matriz_mn_d, matriz_nm_d);

    // Imprimir Transpuesta
    printMatrix("Matriz Transpuesta", m, n, matriz_mn_h);

    // Liberacion de espacios de memoria
    cudaFree(matriz_mn_d);
    cudaFree(matriz_nm_d);

    return 0;
}

int main(int argc, char *argv[])
{
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