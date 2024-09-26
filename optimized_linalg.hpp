#ifndef __OPTIMIZED_LINALG__
#define __OPTIMIZED_LINALG__

#include <string>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace linalg {
	// Matrix
	template<typename T>
	struct Matrix {
		int n_rows, n_cols;
		T *p = nullptr;
	};
	
	template<typename T>
	static bool isMatAllocated(Matrix<T> &mat) {
		// Check if matrix is allocated
		if (mat.p == nullptr) {
			fprintf(stderr, "Error: Matrix is not allocated. Please use linalg::mallocMat() to allocate memory for the matrix.\n");
			exit(EXIT_FAILURE);
			return false;
		}
		return true;
	}

	template<typename T>
	void mallocMat(Matrix<T> &mat, int n_rows, int n_cols) {
		mat.p = (T*)malloc(sizeof(T) * n_rows * n_cols);
		mat.n_rows = n_rows;
		mat.n_cols = n_cols;
	}

	template<typename T>
	void createZeroMat(Matrix<T> &mat) {
		if (isMatAllocated<T>(mat)) {
			for (int i = 0; i < mat.n_rows; i++) {
				for (int j = 0; j < mat.n_cols; j++) {
					mat.p[i * mat.n_cols + j] = 0;
				}
			}
		}
	}
	
	template<typename T>
	void createIdentityMat(Matrix<T> &mat) {
		if (isMatAllocated<T>(mat)) {
			if (mat.n_rows != mat.n_cols) {
				fprintf(stderr, "Error: cannot create an identity matrix out of a non-square matrix (%d, %d).\n", mat.n_rows, mat.n_cols);
				exit(EXIT_FAILURE);
			}
			for (int i = 0; i < mat.n_rows; i++) {
				for (int j = 0; j < mat.n_cols; j++) {
					mat.p[i * mat.n_cols + j] = 0;
					if (i == j) {
						mat.p[i * mat.n_cols + j] = 1;
					}
				}
			}
		}
	}
	
	template<typename T>
	void populateMatWithValues(Matrix<T> &mat, T *vals, int vals_size) {
		if (isMatAllocated<T>(mat)) {
			if ((vals_size) != mat.n_rows * mat.n_cols) {
				fprintf(stderr, "Error: %d values is not enough to populate %dx%d entries of the matrix.\n", vals_size, mat.n_rows, mat.n_cols);
				exit(EXIT_FAILURE);
			}
			for (int i = 0; i < vals_size; i++) {
				mat.p[i] = vals[i];
			}
		}
	}

	template<typename T>
	void printMat(Matrix<T> &mat, std::string name) {
		if (isMatAllocated<T>(mat)) {
			printf("%s\n", name.c_str());
			for (int i = 0; i < mat.n_rows; i++) {
				for (int j = 0; j < mat.n_cols; j++) {
					printf("%f, ", mat.p[i * mat.n_cols + j]);
				}
				printf("\n");
			}
		}
	}
	// Matrix arithmetic
	template<typename T>
	void matCopy(Matrix<T> &dst, Matrix<T> &src) {
		if (isMatAllocated<T>(dst) && isMatAllocated<T>(src)) {
			// Dimension check
			if ((dst.n_rows != dst.n_rows) || (src.n_cols != dst.n_cols)) {
				fprintf(stderr, "Error: Dimension mismatch. A matrix of size (%d, %d) cannot be assigned to a matrix of size (%d, %d).\n", 
						src.n_rows, src.n_cols, dst.n_rows, dst.n_cols);
				exit(EXIT_FAILURE);
			}

			memcpy(dst.p, src.p, src.n_rows * src.n_cols * sizeof(T));
		}
	}

	template<typename T>
	void matAdd(Matrix<T> &matA, Matrix<T> &matB, Matrix<T> &matC) {
		if (isMatAllocated<T>(matA) && isMatAllocated<T>(matB) && isMatAllocated<T>(matC)) {
			// Dimension check
			if ((matA.n_rows != matB.n_rows) || (matA.n_cols != matB.n_cols)) {
				fprintf(stderr, "Error: Dimension mismatch. Matrices of size (%d, %d) and (%d, %d) cannot be added together.\n", 
						matA.n_rows, matA.n_cols, matB.n_rows, matB.n_cols);
				exit(EXIT_FAILURE);
			}
			for (int i = 0; i < matA.n_rows * matA.n_cols; i++) {
					matC.p[i] = matA.p[i] + matB.p[i];
			}
		}
	}

	template<typename T>
	void matScalarMul(Matrix<T> &mat, float scalar, Matrix<T> &result) {
		if (isMatAllocated<T>(mat) && isMatAllocated<T>(result)) {
			for (int i = 0; i < mat.n_rows * mat.n_cols; i++) {
				result.p[i] = mat.p[i] * scalar;
			}
		}
	}

	template<typename T>
	void naiveMatMul(Matrix<T> &matA, Matrix<T> &matB, Matrix<T> &matC) {
		if (isMatAllocated<T>(matA) && isMatAllocated<T>(matB) && isMatAllocated<T>(matC)) {
			// Make sure the matrix multiplication is valid
			if (matA.n_cols != matB.n_rows) {
				fprintf(stderr, "Error: Dimension mismatch. Matrix of size (%d, %d) cannot multiply with matrix of size (%d, %d).\n",
						matA.n_rows ,matA.n_cols, matB.n_rows, matB.n_cols);
				exit(EXIT_FAILURE);
			}
			if (matC.n_rows != matA.n_rows || matC.n_cols != matB.n_cols) {
				fprintf(stderr, "Error: the multiplication between matrices of size (%d, %d) and (%d, %d) should result in matrix of size (%d, %d), instead of (%d, %d).\n", matA.n_rows, matA.n_cols, matB.n_rows, matB.n_cols, matA.n_rows, matB.n_cols, matC.n_rows, matC.n_cols);
				exit(EXIT_FAILURE);
			}
			
		  	// these are the columns A
			float32x4_t     A0;
		  	float32x4_t     A1;
		  	float32x4_t     A2;
		  	float32x4_t     A3;

		  	// these are the columns B
		  	float32x4_t     B0;
		  	float32x4_t     B1;
		  	float32x4_t     B2;
		  	float32x4_t     B3;

		  	// these are the columns C
		  	float32x4_t     C0;
		  	float32x4_t     C1;
		  	float32x4_t     C2;
		  	float32x4_t     C3;

		  	A0 = vld1q_f32(matA.p);
		  	A1 = vld1q_f32(matA.p + 4);
		  	A2 = vld1q_f32(matA.p + 8);
		  	A3 = vld1q_f32(matA.p + 12);


		  	// Zero accumulators for C values
		  	C0 = vmovq_n_f32(0);
		  	C1 = vmovq_n_f32(0);
		  	C2 = vmovq_n_f32(0);
		  	C3 = vmovq_n_f32(0);

		  	// Multiply accumulate in 4x1 blocks, i.e. each column in C
		  	B0 = vld1q_f32(matB.p);
		  	C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
		  	C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
		  	C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
		  	C0 = vfmaq_laneq_f32(C0, A3, B0, 3);
		  	vst1q_f32(matC.p, C0);

		  	B1 = vld1q_f32(matB.p + 4);
		  	C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
		  	C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
		  	C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
		  	C1 = vfmaq_laneq_f32(C1, A3, B1, 3);
		  	vst1q_f32(matC.p + 4, C1);

		  	B2 = vld1q_f32(matB.p + 8);
		  	C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
		  	C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
		  	C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
		  	C2 = vfmaq_laneq_f32(C2, A3, B2, 3);
		  	vst1q_f32(matC.p + 8, C2);

		  	B3 = vld1q_f32(matB.p + 12);
		  	C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
		  	C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
		  	C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
		  	C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
		  	vst1q_f32(matC.p + 12, C3);
		}
	}

	template<typename T>
	Matrix<T> matTranspose(Matrix<T> mat) {
		Matrix<T> mat_T;
		if (isMatAllocated<T>(mat)) {
			mallocMat<T>(mat_T, mat.n_cols, mat.n_rows);
			for (int i = 0; i < mat_T.n_rows; i++) {
				for (int j = 0; j < mat_T.n_cols; j++) {
					mat_T.p[i * mat_T.n_cols + j] = mat.p[j * mat.n_cols + i];
				}
			}
		}
		return mat_T;
	}

	template<typename T>
	void constructTransformationMatrix(Matrix<T> &R, Matrix<T> &p, Matrix<T> &Tr) {
	
		if (isMatAllocated<T>(R) && isMatAllocated<T>(p) && isMatAllocated<T>(Tr)) {
			// TODO: check if R ∈ SO(3)
			// Dimension check
			if ((R.n_rows != 3) || (R.n_cols != 3)) {
				fprintf(stderr, "Error: the rotation part's size is (%d, %d), which is not a 3x3 matrix.\n", R.n_rows, R.n_cols);
				exit(EXIT_FAILURE);

			}

			if ((p.n_rows != 3) || (p.n_cols != 1)) {
				fprintf(stderr, "Error: the translation part's size needs to be (3, 1) instead of (%d, %d).\n", p.n_rows, p.n_cols);
				exit(EXIT_FAILURE);
			}

			if ((Tr.n_rows != 4) || (Tr.n_cols != 4)) {
				fprintf(stderr, "Error: the size of the transformation matrix should be (4, 4) instead of (%d, %d).\n", Tr.n_rows, Tr.n_cols);
				exit(EXIT_FAILURE);
			}

			// Fill in the rotation part with R
			for (int i = 0; i < (Tr.n_rows - 1); i++) {
				for (int j = 0; j < (Tr.n_cols - 1); j++) {
					Tr.p[i * Tr.n_cols + j] = R.p[i * R.n_cols + j];
				}
			}
			// Fill in the translation part with p
			for (int i = 0; i < (Tr.n_rows - 1); i++) {
				Tr.p[i * Tr.n_cols + 3] = p.p[i];
			}
			// Fill in the final row with (0, 0, 0, 1)
			for (int i = 0; i < Tr.n_cols; i++) {
				Tr.p[3 * 4 + i] = 0;
				if (i == 3) {
					Tr.p[3 * 4 + i] = 1;
				}
			}
		}
	}
	// Vector arithmetic
	template<typename T>
	void crossProduct(Matrix<T> &matA, Matrix<T> &matB, Matrix<T> &matC) {
		if (isMatAllocated<T>(matA) && isMatAllocated<T>(matB) && isMatAllocated<T>(matC)) {
			// Check if matrices are 3D vectors
			if ((matA.n_rows != 1) || (matA.n_cols != 3) || 
				(matB.n_rows != 1) || (matB.n_cols != 3) || 
				(matC.n_rows != 1) || (matC.n_cols != 3)) {
				fprintf(stderr, "Error: cannot perform cross product between mathematical objects that are not 3D vectors.\n");
				exit(EXIT_FAILURE);
			}
			for (int i = 0; i < 4; i++) {
				*(matC.p + i - 1) = *(matA.p + (i%3)) * *(matB.p + (i+1)%3) - 
									*(matB.p + (i%3)) * *(matA.p + (i+1)%3);
			}
		}
	}

	template<typename T>
	void convertToSkewSymmetricMatrix(Matrix<T> &vec, Matrix<T> &skew) {
		if (isMatAllocated<T>(vec) && isMatAllocated<T>(skew)) {
			// Check if the matrix is a 3D vector
			if ((vec.n_rows != 1) || (vec.n_cols != 3)) {
				fprintf(stderr, "Error: cannot convert a mathematical object that are not a 3D vector to the skew-symmetric matrix form.\n");
				exit(EXIT_FAILURE);
			}
			if (skew.n_rows != 3 || skew.n_cols != 3) {
				fprintf(stderr, "Error: cannot convert a 3D vector to a skew-symmetric matrix of size (%d, %d).\n", skew.n_rows, skew.n_cols);
				exit(EXIT_FAILURE);
			}
			// Main diagnal of skew-symmetric matrix is zero
			for (int i = 0; i < 3; i++) {
				skew.p[i * 3 + i] = 0;
			}
			skew.p[1] = -vec.p[2];
			skew.p[2] = vec.p[1];
			skew.p[3] = vec.p[2];
			skew.p[5] = -vec.p[0];
			skew.p[6] = -vec.p[1];
			skew.p[7] = vec.p[0];
		}
	}

	// Experiment
	template<typename T>
	void matAddMultiple(Matrix<T> &mat, int n_args, ...) {
		va_list args;
		va_start(args, n_args);
		for (int i = 0; i < n_args; i++) {
			Matrix<T> arg = va_arg(args, Matrix<T>);
			if (isMatAllocated<T>(arg) || isMatAllocated<T>(mat)) {
				if ((arg.n_rows != mat.n_rows) || (arg.n_cols != mat.n_cols)) {
					fprintf(stderr, "Error: dimension mismatch. Cannot perform matrix addition between matrices of size (%d, %d) and (%d, %d).\n", mat.n_rows, mat.n_cols, arg.n_rows, arg.n_cols);
					exit(EXIT_FAILURE);
				}
				for (int i = 0; i < mat.n_rows * mat.n_cols; i++) {
					mat.p[i] += arg.p[i];
				}
			}
		}
		va_end(args);
	}

	template<typename T>
	void gaussianEliminate(Matrix<T> &mat) {
		if (isMatAllocated<T>(mat)) {
			T pivot;
			// iterate over rows of the matrix
			for (int row = 0; row < mat.n_rows; row++) {
				pivot = mat.p[row * mat.n_cols + row];
				// Divide entries behind the pivot by itself
				for (int col = row; col < mat.n_cols; col++) {
					mat.p[row * mat.n_cols + col] /= pivot;
				}
				// Eliminate the pivot col from the remaining rows
				for (int elim_row = row + 1; elim_row < mat.n_rows; elim_row++) {
					// Get the scaling factor for the elimination
					auto scale = mat.p[elim_row * mat.n_cols + row];
					// Remove the pivot 
					for (int col = row; col < mat.n_cols; col++) {
						mat.p[elim_row * mat.n_cols + col] -= mat.p[row * mat.n_cols + col] * scale;
					}
				}
			}
		}
	}

} /*namespace linalg*/

#endif /*__OPTIMIZED_LINALG__*/
