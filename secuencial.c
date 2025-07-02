/*
 * Programa para calcular la pseudo-inversa de una matriz de forma secuencial
 * Autores: Christian Delgado, Rafael Morales
 * Fecha: 10-06-2025
 * Ramo: Computación de Alto Rendimiento
 * Docente: Sergio Antonio Baltierra Valenzuela
 * 
 * Compilar: gcc secuencial.c -o secuencial -lm [entrada]
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

/*
 * Function to get current time in seconds for performance measurement
 */
double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

/**
 * Función que imprime una matriz en un archivo de salida
 * @param label Etiqueta para identificar la matriz en el archivo
 * @param m Número de filas de la matriz
 * @param n Número de columnas de la matriz
 * @param matriz Matriz a imprimir
 */
void printMatrix(char* label, int m, int n, double matriz[m][n]) {    
    FILE* archivo = fopen("salida.sal", "w");

    if (!archivo) {
        perror("Error al abrir el archivo");
        return;
    }

    fprintf(archivo, "%s\n", label);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(archivo, "%.6f ", matriz[i][j]);
        }
        fprintf(archivo, "\n");
    } 

    fclose(archivo);
}

/**
 * Función que calcula el rango de una matriz usando eliminación gaussiana
 * @param m Número de filas de la matriz
 * @param n Número de columnas de la matriz
 * @param A Matriz de entrada
 * @return El rango de la matriz
 * 
 * La función crea una copia de la matriz y aplica eliminación gaussiana
 * para encontrar el rango. Implementación secuencial.
 */
int rango(int m, int n, double A[m][n]) {
    const double EPS = 1e-16;

    // Crear copia de la matriz original
    double temp[m][n];
    int rank = 0;

    // Copiar matriz original a matriz temporal
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            temp[i][j] = A[i][j];
        }
    }

    // Para cada columna
    for (int i = 0; i < n; i++) {
        int pivotRow = -1;

        // Buscar pivote en la columna actual
        for (int j = rank; j < m; j++) {
            if (fabs(temp[j][i]) > EPS) {
                pivotRow = j;
                break;
            }
        }

        // Si se encontró pivote
        if (pivotRow != -1) {
            // Intercambiar fila pivote con fila actual
            for (int k = 0; k < n; k++) {
                double tmp = temp[rank][k];
                temp[rank][k] = temp[pivotRow][k];
                temp[pivotRow][k] = tmp;
            }

            // Reducir las demás filas usando la fila pivote
            for (int j = 0; j < m; j++) {
                if (j != rank) {
                    double factor = temp[j][i] / temp[rank][i];
                    for (int k = 0; k < n; k++)
                        temp[j][k] -= factor * temp[rank][k];
                }
            }

            // Incrementar rango encontrado
            rank++;
        }
    }

    // Retornar rango calculado
    return rank;
}

/**
 * Función que calcula la transpuesta de una matriz
 * @param m Número de filas de la matriz de entrada
 * @param n Número de columnas de la matriz de entrada
 * @param A Matriz de entrada
 * @param B Matriz de salida que contendrá la transpuesta
 * 
 * Implementación secuencial del cálculo de transpuesta.
 */
void transpuesta(int m, int n, double A[m][n], double B[n][m]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B[j][i] = A[i][j];
        }
    }
    return;
}

/**
 * Función que multiplica dos matrices
 * @param m Número de filas de la primera matriz
 * @param n Número de columnas de la primera matriz y filas de la segunda
 * @param p Número de columnas de la segunda matriz
 * @param A Primera matriz
 * @param B Segunda matriz
 * @param C Matriz de salida que contendrá el resultado
 * 
 * Implementación secuencial de la multiplicación de matrices.
 */
void multiplicar(int m, int n, int p, double A[m][n], double B[n][p], double C[m][p]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0;

            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return;
}

/**
 * Función que calcula la inversa de una matriz usando eliminación gaussiana
 * @param m Dimensión de la matriz cuadrada
 * @param A Matriz de entrada
 * @param inv Matriz donde se almacenará la inversa
 * @return 0 si se calcula la inversa exitosamente, 1 si la matriz no es invertible
 * 
 * Implementación secuencial del cálculo de la inversa usando
 * eliminación gaussiana con pivoteo parcial.
 */
int inverse(int m, double A[m][m], double inv[m][m]) {
    const double EPS = 1e-16;

    // Construir matriz aumentada [A | I]
    double aug[m][2 * m];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            aug[i][j] = A[i][j];
            aug[i][j + m] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Eliminar columna por columna
    for (int col = 0, row = 0; col < m && row < m; col++) {
        // Buscar el mejor pivote
        int piv = row;
        for (int k = row + 1; k < m; k++) {
            if (fabs(aug[k][col]) > fabs(aug[piv][col])) {
                piv = k;
            }
        }

        // Verificar pivote
        if (fabs(aug[piv][col]) < EPS) {
            fprintf(stderr, "Error: matriz no invertible (pivote ~ 0)\n");
            return 1;
        }

        // Intercambiar filas
        if (piv != row) {
            for (int j = 0; j < 2 * m; j++) {
                double tmp = aug[piv][j];
                aug[piv][j] = aug[row][j];
                aug[row][j] = tmp;
            }
        }

        // Normalizar fila pivote
        double div = aug[row][col];
        for (int j = 0; j < 2 * m; j++) {
            aug[row][j] /= div;
        }

        // Eliminar columna en otras filas
        for (int i = 0; i < m; i++) {
            if (i == row) continue;
            double factor = aug[i][col];
            for (int j = 0; j < 2 * m; j++) {
                aug[i][j] -= factor * aug[row][j];
            }
        }

        row++;  
    }

    // Copiar parte derecha como inversa
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            inv[i][j] = aug[i][j + m];
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
    // Inicio del temporizador
    //Comando para compilar gcc -fopenmp secuencial.c -o secuencial -lm 
    //Comando para ejecutar: ./secuencial numero_hilos
    // Leer número de hilos desde argumentos, si se proporciona
    double start = get_time();

    // Leer número de hilos desde argumentos, si se proporciona
    int m, n;
    const char *entrada = "entrada.ent";
    FILE* archivo = fopen(entrada, "r");

    if (!archivo) {
        perror("Error al abrir el archivo");
        return 1;
    }

    if (fscanf(archivo, "%d %d", &m, &n) != 2) {
        fprintf(stderr, "Error al leer dimensiones del archivo\n");
        fclose(archivo);
        return 1;
    }

    // Verificar límite de dimensiones
    if (m > 100 || n > 100) {
        FILE* salida = fopen("salida.sal", "w");
        if (salida) {
            fprintf(salida, "Dimensionalidad de la matriz no soportado\n");
            fclose(salida);
        }
        return 1;
    }

    double matriz[m][n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(archivo, "%lf", &matriz[i][j]) != 1) {
                fprintf(stderr, "Error al leer el elemento (%d, %d)\n", i, j);
                fclose(archivo);
                return 1;
            }
        }
    }
    fclose(archivo);

    int r = rango(m, n, matriz);
    if (r < (m < n ? m : n)) {
        FILE* salida = fopen("salida.sal", "w");
        if (salida) {
            fprintf(salida, "-1\n");
            fclose(salida);
        }
        return 1;
    }

    // Pseudoinversa por la derecha: R = (AᵗA)^-1 Aᵗ
    if (r == m && m <= n) {
        double AT[n][m];
        transpuesta(m, n, matriz, AT);

        double ATA[m][m];
        multiplicar(m, n, m, matriz, AT, ATA);

        double invATA[m][m];
        int error = inverse(m, ATA, invATA);
        if (error == 1) {
            FILE* salida = fopen("salida.sal", "w");
            if (salida) {
                fprintf(salida, "-1\n");
                fclose(salida);
            }
            return 1;
        }

        double P[n][m];
        multiplicar(n, m, m, AT, invATA, P);
        printMatrix("R", n, m, P);

    // Pseudoinversa por la izquierda: L = Aᵗ (A Aᵗ)^-1
    } else if (r == n && n < m) {
        double AT[n][m];
        transpuesta(m, n, matriz, AT);

        double AAT[m][m];
        multiplicar(m, n, m, matriz, AT, AAT);

        double invAAT[m][m];
        int error = inverse(m, AAT, invAAT);
        if (error == 1) {
            FILE* salida = fopen("salida.sal", "w");
            if (salida) {
                fprintf(salida, "-1\n");
                fclose(salida);
            }
            return 1;
        }

        double P[n][m];
        multiplicar(n, m, m, AT, invAAT, P);
        printMatrix("L", n, m, P);
    } else {
        FILE* salida = fopen("salida.sal", "w");
        if (salida) {
            fprintf(salida, "-1\n");
            fclose(salida);
        }
        return 1;
    }

    // Fin del temporizador y cálculo del tiempo de ejecución
    double end = get_time();
    double seq_time = end - start;

    //Guardar el tiempo de ejecución secuencial en un archivo para su uso en paralelo.c
    FILE *time_file = fopen("seq_time.txt", "w");
    if (time_file) {
        fprintf(time_file, "%.6f\n", seq_time);
        fclose(time_file);
    } else {
        perror("Error al guardar tiempo secuencial");
    }

    return 0;
}