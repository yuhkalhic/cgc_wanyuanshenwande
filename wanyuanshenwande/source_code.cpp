#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <immintrin.h>

using namespace std;

// 定义一个时间点类型，用于计时
typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

// 定义全局变量
int v_num = 0; // 定点数量
int e_num = 0; // 边数量
// 分别是输入、第一层、第二层的维度
int F0 = 0, F1 = 0, F2 = 0;

//原始图
vector<int> raw_graph;

// CSR
vector<float> csr_val;
vector<int> csr_col;
vector<int> csr_row;

// 定义用于存储特征和权重的指针
float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

// 读取图数据的函数
void readGraph(char *fname) {
  ifstream infile(fname);

  int source;
  int end;

  infile >> v_num >> e_num;

  //raw_graph.resize(e_num * 2);

  while (!infile.eof()) {
    infile >> source >> end;
    if (infile.peek() == EOF) break;
    raw_graph.push_back(source);
    raw_graph.push_back(end);
  }
}

// 原始图转CSR
void raw_graph_to_CSR() {
    csr_val.reserve(e_num);
    csr_col.reserve(e_num);
    csr_row.resize(v_num + 1, 0);

    vector<int> degree(v_num, 0);

    for (int i = 0; i < e_num; i++) {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        degree[src]++;
    }

    for (int i = 0; i < v_num; i++) {
        csr_row[i + 1] = csr_row[i] + degree[i];
    }

    fill(degree.begin(), degree.end(), 0);

    for (int i = 0; i < e_num; i++) {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        int pos = csr_row[src] + degree[src];
        csr_col[pos] = dst;
        degree[src]++;
    }

    for (int i = 0; i < v_num; i++) {
        for (int j = csr_row[i]; j < csr_row[i + 1]; j++) {
            int dst = csr_col[j];
            csr_val[j] = 1 / sqrt(degree[i]) / sqrt(degree[dst]);
        }
    }
}

// 从文件中读取浮点数数组
void readFloat(char *fname, float *&dst, int num) {
  // 32位对齐
  dst = (float *)_mm_malloc(num * sizeof(float), 32);
  FILE *fp = fopen(fname, "rb");
  fread(dst, num * sizeof(float), 1, fp);
  fclose(fp);
}

// 初始化浮点数数组，并将其元素置为0
void initFloat(float *&dst, int num) {
  // 32位对齐
  dst = (float *)_mm_malloc(num * sizeof(float), 32);
  memset(dst, 0, num * sizeof(float));
}

//线程数据结构
struct ThreadData {
    int start;
    int end;
    int in_dim;
    int out_dim;
    float *in_X;
    float *out_X;
    float *W;
    float *W_T;
    int dim;
};

// XW线程函数
void *XW_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    float(*tmp_in_X)[data->in_dim] = (float(*)[data->in_dim])data->in_X;
    float(*tmp_out_X)[data->out_dim] = (float(*)[data->out_dim])data->out_X;
    float(*tmp_W_T)[data->in_dim] = (float(*)[data->in_dim])data->W_T; // 使用转置矩阵

    for (int i = data->start; i < data->end; i++) {
        for (int j = 0; j < data->out_dim; j++) {
            float sum = 0.0;
            int k = 0;
            __m512 sum_vec = _mm512_setzero_ps();
            for (; k <= data->in_dim - 16; k += 16) {
                __m512 x_vec = _mm512_loadu_ps(&tmp_in_X[i][k]);
                __m512 w_vec = _mm512_loadu_ps(&tmp_W_T[j][k]);
                sum_vec = _mm512_fmadd_ps(x_vec, w_vec, sum_vec);
            }
            // 水平加和求得部分总和
            sum += _mm512_reduce_add_ps(sum_vec);
            // 处理剩余的元素
            for (; k < data->in_dim; k++) {
                sum += tmp_in_X[i][k] * tmp_W_T[j][k];
            }
            tmp_out_X[i][j] = sum;
        }
    }
    pthread_exit(NULL);
}

// 执行矩阵乘法操作（特征*权重），使用转置矩阵
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W) {
    // 创建转置矩阵
    float *W_T = new float[out_dim * in_dim];
    for (int i = 0; i < in_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            W_T[j * in_dim + i] = W[i * out_dim + j];
        }
    }

    int num_threads = 56;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int chunk_size = v_num / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? v_num : (i + 1) * chunk_size;
        thread_data[i].in_dim = in_dim;
        thread_data[i].out_dim = out_dim;
        thread_data[i].in_X = in_X;
        thread_data[i].out_X = out_X;
        thread_data[i].W_T = W_T; // 传递转置矩阵
        pthread_create(&threads[i], NULL, XW_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // 释放转置矩阵内存
    delete[] W_T;
}


// 执行矩阵乘法的函数（特征*权重）
void XW2(int in_dim, int out_dim, float *in_X, float *out_X, float *W) {
  float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
  float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
  float *tmp_W_T = new float[out_dim * in_dim];

  // 转置权重矩阵
  for (int i = 0; i < in_dim; i++) {
    for (int j = 0; j < out_dim; j++) {
      tmp_W_T[j * in_dim + i] = W[i * out_dim + j];
    }
  }

  // 使用AVX-512指令集进行矩阵乘法
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < out_dim; j++) {
      float sum = 0.0f;
      int k = 0;
      __m512 sum_vec = _mm512_setzero_ps();
      for (; k <= in_dim - 16; k += 16) {
        __m512 x_vec = _mm512_loadu_ps(&tmp_in_X[i][k]);
        __m512 w_vec = _mm512_loadu_ps(&tmp_W_T[j * in_dim + k]);
        sum_vec = _mm512_fmadd_ps(x_vec, w_vec, sum_vec);
      }
      sum += _mm512_reduce_add_ps(sum_vec); // 将向量中的元素相加
      for (; k < in_dim; k++) { // 处理剩余部分
        sum += tmp_in_X[i][k] * tmp_W_T[j * in_dim + k];
      }
      tmp_out_X[i][j] = sum;
    }
  }
  delete[] tmp_W_T;
}


//AX线程函数
void *AX_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    float(*tmp_in_X)[data->dim] = (float(*)[data->dim])data->in_X;
    float(*tmp_out_X)[data->dim] = (float(*)[data->dim])data->out_X;

    int vec_len = 16;
    int rounded_dim = (data->dim / vec_len) * vec_len;

    for (int i = data->start; i < data->end; i++) {
        for (int j = csr_row[i]; j < csr_row[i + 1]; j++) {
            int col = csr_col[j];
            float weight = csr_val[j];
            int k = 0;
            for (; k < rounded_dim; k += vec_len) {
                __m512 val_vec = _mm512_loadu_ps(&tmp_in_X[col][k]);
                __m512 weight_vec = _mm512_set1_ps(weight);
                __m512 out_vec = _mm512_loadu_ps(&tmp_out_X[i][k]);
                out_vec = _mm512_fmadd_ps(val_vec, weight_vec, out_vec);
                _mm512_storeu_ps(&tmp_out_X[i][k], out_vec);
            }

            for (; k < data->dim; k++) {
                tmp_out_X[i][k] += tmp_in_X[col][k] * weight;
            }
        }
    }
    pthread_exit(NULL);
}

// 使用CSR格式的图执行AX聚合函数的操作（邻接矩阵*特征）
void AX(int dim, float *in_X, float *out_X) {
    memset(out_X, 0, v_num * dim * sizeof(float));

    int num_threads = 56;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int chunk_size = v_num / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? v_num : (i + 1) * chunk_size;
        thread_data[i].dim = dim;
        thread_data[i].in_X = in_X;
        thread_data[i].out_X = out_X;
        pthread_create(&threads[i], NULL, AX_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// 小数据执行聚合函数的操作（邻接矩阵*特征）
void AX2(int dim, float *in_X, float *out_X) {
    memset(out_X, 0, v_num * dim * sizeof(float)); // 清零输出数组
    float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
    float(*tmp_out_X)[dim] = (float(*)[dim])out_X;

    int vec_len = 16; // 每次处理 16 个浮点数
    int rounded_dim = (dim / vec_len) * vec_len; // 确保处理长度为 16 的倍数

    for (int i = 0; i < v_num; i++) {
        for (int j = csr_row[i]; j < csr_row[i+1]; j++) {
            int col = csr_col[j];
            float weight = csr_val[j];
            __m512 weight_vec = _mm512_set1_ps(weight); // 将权重扩展到 16 个浮点数的向量

            int k = 0;
            for (; k < rounded_dim; k += vec_len) {
                __m512 col_vec = _mm512_loadu_ps(&tmp_in_X[col][k]); // 加载输入特征向量
                __m512 out_vec = _mm512_loadu_ps(&tmp_out_X[i][k]); // 加载输出特征向量
                __m512 result_vec = _mm512_fmadd_ps(weight_vec, col_vec, out_vec); // 执行加权累加
                _mm512_storeu_ps(&tmp_out_X[i][k], result_vec); // 存储结果
            }

            // 处理剩余的不足 16 个的元素
            for (; k < dim; k++) {
                tmp_out_X[i][k] += weight * tmp_in_X[col][k];
            }
        }
    }
}

//ReLU线程函数
void *ReLU_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start; i < data->end; i++) {
        if (data->in_X[i] < 0) data->in_X[i] = 0;
    }
    pthread_exit(NULL);
}

// ReLU激活函数
void ReLU(int dim, float *X) {
    int num_threads = 56;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int chunk_size = v_num * dim / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? v_num * dim : (i + 1) * chunk_size;
        thread_data[i].in_X = X;
        pthread_create(&threads[i], NULL, ReLU_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// ReLU激活函数
void ReLU2(int dim, float *X) {
    int total_length = v_num * dim;
    int i = 0;
    __m512 zero_vec = _mm512_setzero_ps(); // 创建一个全0的向量用于比较

    // 以16个float为一组进行处理
    for (; i <= total_length - 16; i += 16) {
        __m512 vec = _mm512_loadu_ps(&X[i]); // 未对齐加载
        vec = _mm512_max_ps(vec, zero_vec);  // 比较并取最大值，实现ReLU功能
        _mm512_storeu_ps(&X[i], vec);        // 存回内存
    }

    // 处理剩余不足16个的部分
    for (; i < total_length; i++) {
        if (X[i] < 0)
            X[i] = 0;
    }
}

//LogSoftmax线程函数
void *LogSoftmax_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    float(*tmp_X)[data->dim] = (float(*)[data->dim])data->in_X;

    for (int i = data->start; i < data->end; i++) {
        float max = tmp_X[i][0];
        for (int j = 1; j < data->dim; j++) {
            if (tmp_X[i][j] > max) max = tmp_X[i][j];
        }

        float sum = 0;
        for (int j = 0; j < data->dim; j++) {
            sum += exp(tmp_X[i][j] - max);
        }
        sum = log(sum);

        for (int j = 0; j < data->dim; j++) {
            tmp_X[i][j] = tmp_X[i][j] - max - sum;
        }
    }
    pthread_exit(NULL);
}

// LogSoftmax归一化函数
void LogSoftmax(int dim, float *X) {
    int num_threads = 56;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int chunk_size = v_num / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? v_num : (i + 1) * chunk_size;
        thread_data[i].dim = dim;
        thread_data[i].in_X = X;
        pthread_create(&threads[i], NULL, LogSoftmax_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

// 小数据LogSoftmax归一化函数
void LogSoftmax2(int dim, float *X) {
  float(*tmp_X)[dim] = (float(*)[dim])X;

  for (int i = 0; i < v_num; i++) {
    float max = tmp_X[i][0];
    for (int j = 1; j < dim; j++) {
      if (tmp_X[i][j] > max) max = tmp_X[i][j];
    }

    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += exp(tmp_X[i][j] - max);
    }
    sum = log(sum);

    for (int j = 0; j < dim; j++) {
      tmp_X[i][j] = tmp_X[i][j] - max - sum;
    }
  }
}

// MaxRowSum线程结构体
struct MaxRowSumData {
    int start;
    int end;
    int dim;
    float *X;
    float max_sum;
};

// MaxRowSum线程函数计算矩阵的最大行和
void *MaxRowSum_thread(void *arg) {
    MaxRowSumData *data = (MaxRowSumData *)arg;
    float(*tmp_X)[data->dim] = (float(*)[data->dim])data->X;
    float max = -__FLT_MAX__;
    int vec_len = 16;
    int rounded_dim = (data->dim / vec_len) * vec_len;

    for (int i = data->start; i < data->end; i++) {
        __m512 sum_vec = _mm512_setzero_ps();  // 初始化求和向量为0
        for (int j = 0; j < rounded_dim; j += vec_len) {
            __m512 row_vec = _mm512_load_ps(&tmp_X[i][j]);
            sum_vec = _mm512_add_ps(sum_vec, row_vec);
        }
        float sum_array[16];
        _mm512_store_ps(sum_array, sum_vec);
        float sum = 0;
        for (int k = 0; k < 16; k++) {
            sum += sum_array[k];
        }
        for (int j = rounded_dim; j < data->dim; j++) {
            sum += tmp_X[i][j];
        }
        if (sum > max) {
            max = sum;
        }
    }
    data->max_sum = max;
    pthread_exit(NULL);
}

// MaxRowSum函数计算矩阵的最大行和
float MaxRowSum(float *X, int dim) {
    int num_threads = 56;
    pthread_t threads[num_threads];
    MaxRowSumData thread_data[num_threads];
    int chunk_size = v_num / num_threads;
    float global_max = -__FLT_MAX__;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? v_num : (i + 1) * chunk_size;
        thread_data[i].dim = dim;
        thread_data[i].X = X;
        pthread_create(&threads[i], NULL, MaxRowSum_thread, (void *)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        if (thread_data[i].max_sum > global_max) {
            global_max = thread_data[i].max_sum;
        }
    }

    return global_max;
}

//小数据MaxRowSum函数计算矩阵的最大行和
float MaxRowSum2(float *X, int dim) {
    float(*tmp_X)[dim] = (float(*)[dim])X;
    float global_max = -__FLT_MAX__;  // 初始化全局最大值为最小浮点数

    for (int i = 0; i < v_num; i++) {
        __m512 sum_vec = _mm512_setzero_ps();  // 使用 AVX512 初始化求和向量为0

        // 向量化处理大部分元素
        int j = 0;
        for (; j <= dim - 16; j += 16) {
            __m512 row_vec = _mm512_loadu_ps(&tmp_X[i][j]);  // 加载行数据
            sum_vec = _mm512_add_ps(sum_vec, row_vec);       // 累加向量
        }

        // 使用水平加法将向量中的值求和
        __m256 low_half = _mm512_extractf32x8_ps(sum_vec, 0);
        __m256 high_half = _mm512_extractf32x8_ps(sum_vec, 1);
        __m256 sum_vec256 = _mm256_add_ps(low_half, high_half);
        __m128 lower = _mm256_extractf128_ps(sum_vec256, 0);
        __m128 higher = _mm256_extractf128_ps(sum_vec256, 1);
        __m128 sum_vec128 = _mm_add_ps(lower, higher);
        sum_vec128 = _mm_hadd_ps(sum_vec128, sum_vec128);
        sum_vec128 = _mm_hadd_ps(sum_vec128, sum_vec128);

        float sum = _mm_cvtss_f32(sum_vec128);  // 从 __m128 中提取最终求和结果

        // 处理剩余元素
        for (; j < dim; j++) {
            sum += tmp_X[i][j];
        }

        // 更新全局最大和
        if (sum > global_max) {
            global_max = sum;
        }
    }
    return global_max;
}

// 图数据预处理函数
void somePreprocessing() {
  raw_graph_to_CSR();
}

// 释放内存
void freeFloats() {
  _mm_free(X0);
  _mm_free(W1);
  _mm_free(W2);
  _mm_free(X1);
  _mm_free(X2);
  _mm_free(X1_inter);
  _mm_free(X2_inter);
}

int main(int argc, char **argv) {
    F0 = atoi(argv[1]);
    F1 = atoi(argv[2]);
    F2 = atoi(argv[3]);

    readGraph(argv[4]);
    readFloat(argv[5], X0, v_num * F0);
    readFloat(argv[6], W1, F0 * F1);
    readFloat(argv[7], W2, F1 * F2);

    initFloat(X1, v_num * F1);
    initFloat(X1_inter, v_num * F1);
    initFloat(X2, v_num * F2);
    initFloat(X2_inter, v_num * F2);

    TimePoint start = chrono::steady_clock::now();

    somePreprocessing();
    float max_sum;
    if(v_num  < 500000 && e_num  < 500000){
        XW2(F0, F1, X0, X1_inter, W1);
        AX2(F1, X1_inter, X1);
        ReLU2(F1, X1);
        XW2(F1, F2, X1, X2_inter, W2);
        AX2(F2, X2_inter, X2);
        LogSoftmax2(F2, X2);

        max_sum = MaxRowSum2(X2, F2);
    }else{
        XW(F0, F1, X0, X1_inter, W1);
        AX(F1, X1_inter, X1);
        ReLU(F1, X1);
        XW(F1, F2, X1, X2_inter, W2);
        AX(F2, X2_inter, X2);
        LogSoftmax(F2, X2);

        max_sum = MaxRowSum(X2, F2);
    }

    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    printf("%.8f\n", max_sum);
    printf("%.8lf\n", l_timeMs);

    freeFloats();
    return 0;
}