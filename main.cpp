#include <string>
#include <omp.h>
#include <cmath>
#include <benchmark/benchmark.h>
#include "linalg.hpp" // header-only library
#include <mpi.h>

#define VECTOR_SIZE 3

template<typename T>
void PoE(T *thetas, T *points, T *omegas, linalg::Matrix<T> &result, int N);

typedef struct RoboticArmSpecs {
	const int N_JOINTS = 4;
	const float L1 = 31.0f;
	const float L2 = 80.0f;
	const float L3 = 80.0f;
	const float L4 = 62.0f;
} RoboticArmSpecs;

static void poe_bench(benchmark::State &s) {
	RoboticArmSpecs arm;
	// Matrices declaration
	linalg::Matrix<float> M, T_eb, result;
	// Matrices allocation
	linalg::mallocMat<float>(M, 4, 4);
	linalg::mallocMat<float>(T_eb, 4, 4);
	linalg::mallocMat<float>(result, 4, 4);
	// Matrices initialization
	float M_vals[] = {0, 0, 1, 0,
					  1, 0, 0, 0,
					  0, 1, 0, arm.L1 + arm.L2 + arm.L3,
					  0, 0, 0, 1};
	linalg::populateMatWithValues<float>(M, M_vals, sizeof(M_vals)/sizeof(float));
	// PoE parameters
	// Joint angles
	float thetas[arm.N_JOINTS] = {0, 0, M_PI/2, 0};
	// Random points on screw axes
	float points[arm.N_JOINTS * VECTOR_SIZE] = {0, 0, 0,
									  			0, 0, arm.L1,
									  			0, 0, arm.L1 + arm.L2,
									  			0, 0, arm.L1 + arm.L2 + arm.L3};
	// angular velocities
	float omegas[arm.N_JOINTS * VECTOR_SIZE] = {0, 0, 1,
									  			1, 0, 0,
												1, 0, 0,
												1, 0, 0};
	for (auto _: s) {
		PoE<float>(thetas, points, omegas, T_eb, arm.N_JOINTS);
		linalg::naiveMatMul<float>(T_eb, M, result);
		//for (int i = 0; i < 100000; i++) {
		//	linalg::createIdentityMat<float>(T_eb);
		//}
	}
	linalg::printMat<float>(result, "Final result");
}

BENCHMARK(poe_bench)->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();

template<typename T>
void PoE(T *thetas, T *points, T *omegas, linalg::Matrix<T> &result, int N) {
	// create tmp to store accumulative product of exponentials
	linalg::Matrix<T> tmp;
	linalg::mallocMat<T>(tmp, 4, 4);
	linalg::createIdentityMat<T>(tmp);

	// PoE algorithm
	for (int i = 0; i < N; i++) {
		// printf("============ i = %d =============\n", i);
		// Create a vector omega at each joint
		linalg::Matrix<T> omega, skew_omega;
		linalg::mallocMat<T>(omega, 1, 3);
		linalg::mallocMat<T>(skew_omega, 3, 3);
		// Populate the vector omega
		T omega_vals[VECTOR_SIZE];
		for (int j = 0; j < VECTOR_SIZE; j++) {
			omega_vals[j] = omegas[i * VECTOR_SIZE + j];
		}
		linalg::populateMatWithValues<T>(omega, omega_vals, sizeof(omega_vals)/sizeof(T));
		// Convert vector omega to its skew-symmetric matrix form
		linalg::convertToSkewSymmetricMatrix<T>(omega, skew_omega);
		// linalg::printMat<T>(skew_omega, "skew omega");
		// Create a linear velocity vector
		linalg::Matrix<T> v, point;
		linalg::mallocMat<T>(v, 1, 3);
		linalg::mallocMat<T>(point, 1, 3);
		// Populate the vector point
		T point_vals[VECTOR_SIZE];
		for (int j = 0; j < VECTOR_SIZE; j++) {
			point_vals[j] = points[i * VECTOR_SIZE + j];
		}
		linalg::populateMatWithValues<T>(point, point_vals, sizeof(point_vals)/sizeof(T));
		// Calculate the linear velocity v = - omega x point
		linalg::crossProduct<T>(omega, point, v);
		linalg::matScalarMul<T>(v, -1.0f, v);

		// Calculate the rotation part of the exponential
		linalg::Matrix<T> R, R_term_1, R_term_2, R_term_3;
		linalg::mallocMat<T>(R, 3, 3);
		linalg::mallocMat<T>(R_term_1, 3, 3);
		linalg::mallocMat<T>(R_term_2, 3, 3);
		linalg::mallocMat<T>(R_term_3, 3, 3);
		// Compute R_term_1 = I
		linalg::createIdentityMat<T>(R_term_1);
		// linalg::printMat<T>(R_term_1, "R_term_1");
		// Compute R_term_2 = skew_omega * sin(theta)
		linalg::matScalarMul<T>(skew_omega, sin(thetas[i]), R_term_2);
		// linalg::printMat<T>(R_term_2, "R_term_2");
		// Compute R_term_3 = skew_omega^2 * (1-cos(theta))
		linalg::naiveMatMul<T>(skew_omega, skew_omega, R_term_3);
		linalg::matScalarMul<T>(R_term_3, (1 - cos(thetas[i])), R_term_3);
		// linalg::printMat<T>(R_term_3, "R_term_3");
		// Compute R = R_term_1 + R_term_2 + R_term_3
		linalg::createZeroMat<T>(R);
		linalg::matAddMultiple<T>(R, 3, R_term_1, R_term_2, R_term_3);
		// linalg::printMat<T>(R, "rotation part");

		// Calculate the translation part of the exponential
		linalg::Matrix<T> p, p_sum, p_term_1, p_term_2, p_term_3;
		linalg::mallocMat<T>(p, 3, 1);
		linalg::mallocMat<T>(p_sum, 3, 3);
		linalg::mallocMat<T>(p_term_1, 3, 3);
		linalg::mallocMat<T>(p_term_2, 3, 3);
		linalg::mallocMat<T>(p_term_3, 3, 3);
		// Compute p_term_1 = I * theta
		linalg::createIdentityMat<T>(p_term_1);
		linalg::matScalarMul<T>(p_term_1, thetas[i], p_term_1);
		// Compute p_term_2 = (1-cos(theta)) * skew_omega
		linalg::matScalarMul<T>(skew_omega, (1 - cos(thetas[i])), p_term_2);
		// Compute p_term_3 = (theta - sin(theta)) * skew_omega^2
		linalg::naiveMatMul<T>(skew_omega, skew_omega, p_term_3);
		linalg::matScalarMul<T>(p_term_3, (thetas[i] - sin(thetas[i])), p_term_3);
		// Compute p_sum = p_term_1 + p_term_2 + p_term_3
		linalg::createZeroMat<T>(p_sum);
		linalg::matAddMultiple<T>(p_sum, 3, p_term_1, p_term_2, p_term_3);
		// linalg::printMat<T>(p_sum, "translation part matrix");
		// Compute p = p_sum * v_T
		// linalg::printMat<T>(v, "v");
		linalg::Matrix<T> v_T;
		linalg::mallocMat<T>(v_T, v.n_cols, v.n_rows);
		v_T = linalg::matTranspose<T>(v);
		linalg::naiveMatMul<T>(p_sum, v_T, p);
		// linalg::printMat<T>(p, "translation part");

		// Combine R and p into a homogeneous matrix
		// Create an exponential at each iteration
		linalg::Matrix<T> exp_twist_theta;
		linalg::mallocMat<T>(exp_twist_theta, 4, 4);
		linalg::constructTransformationMatrix<T>(R, p, exp_twist_theta);
		// linalg::printMat<T>(exp_twist_theta, "Transformation matrix");
		// TODO: Calculate the product of exponentials
		// linalg::printMat<T>(tmp, "tmp");
		linalg::naiveMatMul<T>(tmp, exp_twist_theta, result);
		// linalg::printMat<T>(result, "PoE");
		linalg::matCopy<T>(tmp, result);
	}
}

