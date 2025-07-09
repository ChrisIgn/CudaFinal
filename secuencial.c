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

void printMatrix(char* label, int m, int n, double** matriz) {
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

double** reservar_matriz(int m, int n) {
    double** matriz = (double**)malloc(m * sizeof(double*));
    for (int i = 0; i < m; i++) {
        matriz[i] = (double*)malloc(n * sizeof(double));
    }
    return matriz;
}

void liberar_matriz(double** matriz, int m) {
    for (int i = 0; i < m; i++) {
        free(matriz[i]);
    }
    free(matriz);
}

int rango(int m, int n, double** A) {
    const double EPS = 1e-16;
    double** temp = reservar_matriz(m, n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            temp[i][j] = A[i][j];

    int rank = 0;
    for (int i = 0; i < n; i++) {
        int pivotRow = -1;
        for (int j = rank; j < m; j++) {
            if (fabs(temp[j][i]) > EPS) {
                pivotRow = j;
                break;
            }
        }
        if (pivotRow != -1) {
            for (int k = 0; k < n; k++) {
                double tmp = temp[rank][k];
                temp[rank][k] = temp[pivotRow][k];
                temp[pivotRow][k] = tmp;
            }
            for (int j = 0; j < m; j++) {
                if (j != rank) {
                    double factor = temp[j][i] / temp[rank][i];
                    for (int k = 0; k < n; k++)
                        temp[j][k] -= factor * temp[rank][k];
                }
            }
            rank++;
        }
    }
    liberar_matriz(temp, m);
    return rank;
}

void transpuesta(int m, int n, double** A, double** B) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            B[j][i] = A[i][j];
}

void multiplicar(int m, int n, int p, double** A, double** B, double** C) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

int inverse(int m, double** A, double** inv) {
    const double EPS = 1e-16;
    double** aug = reservar_matriz(m, 2 * m);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            aug[i][j] = A[i][j];
            aug[i][j + m] = (i == j) ? 1.0 : 0.0;
        }

    for (int col = 0, row = 0; col < m && row < m; col++) {
        int piv = row;
        for (int k = row + 1; k < m; k++)
            if (fabs(aug[k][col]) > fabs(aug[piv][col]))
                piv = k;

        if (fabs(aug[piv][col]) < EPS) {
            liberar_matriz(aug, m);
            return 1;
        }

        if (piv != row) {
            for (int j = 0; j < 2 * m; j++) {
                double tmp = aug[piv][j];
                aug[piv][j] = aug[row][j];
                aug[row][j] = tmp;
            }
        }

        double div = aug[row][col];
        for (int j = 0; j < 2 * m; j++)
            aug[row][j] /= div;

        for (int i = 0; i < m; i++) {
            if (i == row) continue;
            double factor = aug[i][col];
            for (int j = 0; j < 2 * m; j++)
                aug[i][j] -= factor * aug[row][j];
        }
        row++;
    }

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            inv[i][j] = aug[i][j + m];

    liberar_matriz(aug, m);
    return 0;
}

int main(int argc, char* argv[]) {
    double start = get_time();

    int m, n;
    FILE* archivo = fopen("entrada_grande.ent", "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return 1;
    }

    if (fscanf(archivo, "%d %d", &m, &n) != 2) {
        fclose(archivo);
        return 1;
    }

    double** matriz = reservar_matriz(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            fscanf(archivo, "%lf", &matriz[i][j]);

    fclose(archivo);

    int r = rango(m, n, matriz);
    if (r < (m < n ? m : n)) {
        FILE* salida = fopen("salida.sal", "w");
        if (salida) {
            fprintf(salida, "-1\n");
            fclose(salida);
        }
        liberar_matriz(matriz, m);
        return 1;
    }

    if (r == m && m <= n) {
        double** AT = reservar_matriz(n, m);
        transpuesta(m, n, matriz, AT);

        double** ATA = reservar_matriz(m, m);
        multiplicar(m, n, m, matriz, AT, ATA);

        double** invATA = reservar_matriz(m, m);
        if (inverse(m, ATA, invATA) == 1) {
            FILE* salida = fopen("salida.sal", "w");
            if (salida) {
                fprintf(salida, "-1\n");
                fclose(salida);
            }
            return 1;
        }

        double** P = reservar_matriz(n, m);
        multiplicar(n, m, m, AT, invATA, P);
        printMatrix("R", n, m, P);

        liberar_matriz(AT, n);
        liberar_matriz(ATA, m);
        liberar_matriz(invATA, m);
        liberar_matriz(P, n);

    } else if (r == n && n < m) {
        double** AT = reservar_matriz(n, m);
        transpuesta(m, n, matriz, AT);

        double** AAT = reservar_matriz(m, m);
        multiplicar(m, n, m, matriz, AT, AAT);

        double** invAAT = reservar_matriz(m, m);
        if (inverse(m, AAT, invAAT) == 1) {
            FILE* salida = fopen("salida.sal", "w");
            if (salida) {
                fprintf(salida, "-1\n");
                fclose(salida);
            }
            return 1;
        }

        double** P = reservar_matriz(n, m);
        multiplicar(n, m, m, AT, invAAT, P);
        printMatrix("L", n, m, P);

        liberar_matriz(AT, n);
        liberar_matriz(AAT, m);
        liberar_matriz(invAAT, m);
        liberar_matriz(P, n);
    }

    liberar_matriz(matriz, m);

    double end = get_time();
    double seq_time = end - start;

    FILE* time_file = fopen("seq_time.txt", "w");
    if (time_file) {
        fprintf(time_file, "%.6f\n", seq_time);
        fclose(time_file);
    }

    return 0;
}
