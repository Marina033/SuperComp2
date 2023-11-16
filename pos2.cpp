#include <iostream>
#include <vector>
#include <chrono>


// Функция для вычисления значения k(x, y)
double k(double x, double y, double tol=0.0001) {
  if (x >= -3 && x <= 3 && y >= 0 && y <= 3) {
    return 1;
  } else {
    return 1 / tol;
  }
}

// Функция для вычисления значения F(x, y)
double F(double x, double y) {
  if (x >= -3 && x <= 3 && y >= 0 && y <= 3) {
    return 1;
  } else {
    return 0;
  }
}

// Функция для решения дифференциального уравнения на прямоугольнике Π
void solveEquation(int N,double epsilon, double A1, double A2, double B1, double B2) {
  auto start = std::chrono::high_resolution_clock::now();
  // Определение параметров сетки
  int Nx = N/* ваше значение */;
  int Ny = N/* ваше значение */;

  // Шаг сетки
  double hx = (B1 - A1) / (Nx - 1);
  double hy = (B2 - A2) / (Ny - 1);

  // Создание сетки для хранения решения v
  std::vector<std::vector<double>> v(Nx, std::vector<double>(Ny, 0.0));

  // Итерационный процесс для решения уравнения
  double error = 1000000/* начальное */;
  double tolerance = epsilon/* ваше значение */;
  int maxIterations = 10000 /* ваше значение */;
  int iteration = 0;
  double vOld = 0.0;

  while (error > tolerance && iteration < maxIterations) {
    error = 0.0;

    // Обновление значений внутренних узлов сетки
    for (int i = 1; i < Nx - 1; ++i) {
      for (int j = 1; j < Ny - 1; ++j) {
        double vOld = v[i][j];
        v[i][j] = (k(A1 + i * hx, A2 + j * hy) * (v[i - 1][j] + v[i + 1][j]) / (hx * hx) +
                   k(A1 + i * hx, A2 + j * hy) * (v[i][j - 1] + v[i][j + 1]) / (hy * hy) -
                   F(A1 + i * hx, A2 + j * hy)) /
                  (2 * k(A1 + i * hx, A2 + j * hy) / (hx * hx) + 2 * k(A1 + i * hx, A2 + j * hy) / (hy * hy));
        error = std::max(error, std::abs(v[i][j] - vOld));
      }
    }

    // Граничные условия
    for (int i = 0; i < Nx; ++i) {
      v[i][0] = 0.0;
      v[i][Ny - 1] = 0.0;
    }
    for (int j = 1; j < Ny - 1; ++j) {
      v[0][j] = 0.0;
      v[Nx - 1][j] = 0.0;
    }

    // Вычисление ошибки и инкрементирование счетчика итераций
    iteration++;
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Вычисление времени выполнения в миллисекундах
  std::chrono::duration<double, std::milli> duration = end - start;
  double executionTime = duration.count();

  // Вывод времени выполнения
  std::cout << "Time to solveEquation no-parallel: " << executionTime << " ms" << std::endl;
  std::cout << "Result after " << iteration << " iterations" << std::endl;

  // Вывод результата
  std::cout << "Results: "<<std::endl;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      std::cout << v[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

int main() {
  double A1 = -4, B1 = 4, A2 = -1, B2 = 3; //rectangle
  double epsilon = 1e-6;
  int N = 21;


  solveEquation(N, epsilon, A1, A2, B1, B2);

  return 0;
}