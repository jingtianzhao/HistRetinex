#include "mex.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

// MATLAB column-major index for 256x256 matrix, 0-based
static inline mwSize IDX2(mwSize r, mwSize c) {
    return r + c * 256;
}

static void checkVector256(const mxArray* arr, const char* name) {
    if (!mxIsDouble(arr) || mxIsComplex(arr)) {
        mexErrMsgIdAndTxt("fun_for_hrm_core_mex:InputType", "%s must be a real double vector.", name);
    }
    if (mxGetNumberOfElements(arr) != 256) {
        mexErrMsgIdAndTxt("fun_for_hrm_core_mex:InputSize", "%s must contain exactly 256 elements, e.g., 256x1.", name);
    }
}

/*
 * Core MEX function for fun_for_hrm.
 *
 * Usage from MATLAB:
 * [L_new,new_L,R,result2,iter] = fun_for_hrm_core_mex(N1,C1,L_init,L6,L5,gamma,P,alpha,beta,maxIter,tolL,tolR)
 *
 * Inputs:
 *   N1      : 256x1 histogram counts of V channel, from imhist
 *   C1      : 256x1 bin locations, from imhist
 *   L_init  : 256x1 histogram of mystand(V), from imhist(mystand(...))
 *   L6      : 256x1 external vector, corresponding to temp in original code
 *   L5      : 256x1 external vector, corresponding to temp2 in original code
 *   gamma   : scalar
 *   P       : scalar
 *   alpha   : scalar
 *   beta    : scalar
 *   maxIter : scalar, maximum number of iterations, original code uses iter > 3000
 *   tolL    : scalar, original code uses 0.1
 *   tolR    : scalar, original code uses 0.1
 *
 * Outputs:
 *   L_new   : 256x1
 *   new_L   : 256x1
 *   R       : 256x1
 *   result2 : 256x1
 *   iter    : scalar, actual ending iteration index
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 10 || nrhs > 12) {
        mexErrMsgIdAndTxt("fun_for_hrm_core_mex:NumInputs",
            "Usage: [L_new,new_L,R,result2,iter] = fun_for_hrm_core_mex(N1,C1,L_init,L6,L5,gamma,P,alpha,beta,maxIter,tolL,tolR)");
    }
    if (nlhs < 4 || nlhs > 5) {
        mexErrMsgIdAndTxt("fun_for_hrm_core_mex:NumOutputs", "This function returns 4 or 5 outputs.");
    }

    checkVector256(prhs[0], "N1");
    checkVector256(prhs[1], "C1");
    checkVector256(prhs[2], "L_init");
    checkVector256(prhs[3], "L6");
    checkVector256(prhs[4], "L5");

    const double* N1 = mxGetPr(prhs[0]);
    const double* C1 = mxGetPr(prhs[1]);
    const double* L_init = mxGetPr(prhs[2]);
    const double* L6 = mxGetPr(prhs[3]);
    const double* L5 = mxGetPr(prhs[4]);

    const double gamma = mxGetScalar(prhs[5]);
    const double P     = mxGetScalar(prhs[6]);
    const double alpha = mxGetScalar(prhs[7]);
    const double beta  = mxGetScalar(prhs[8]);
    const int maxIter  = static_cast<int>(mxGetScalar(prhs[9]));
    const double tolL  = (nrhs >= 11) ? mxGetScalar(prhs[10]) : 0.1;
    const double tolR  = (nrhs >= 12) ? mxGetScalar(prhs[11]) : 0.1;

    if (!(maxIter >= 1)) {
        mexErrMsgIdAndTxt("fun_for_hrm_core_mex:BadMaxIter", "maxIter must be >= 1.");
    }
    if (gamma == 0.0) {
        mexErrMsgIdAndTxt("fun_for_hrm_core_mex:BadGamma", "gamma must not be 0.");
    }

    // N = sum(N1), which equals m*n in the original function under normal imhist usage.
    double N = 0.0;
    for (mwSize i = 0; i < 256; ++i) N += N1[i];
    if (N <= 0.0) {
        mexErrMsgIdAndTxt("fun_for_hrm_core_mex:BadHistogram", "sum(N1) must be positive.");
    }

    // E_C = C1.^P; B = E_C * E_C';
    std::vector<double> E_C(256);
    for (mwSize i = 0; i < 256; ++i) E_C[i] = std::pow(C1[i], P);

    // index, stores 0-based indices, equivalent to MATLAB 1..256 minus 1.
    std::vector<int> index(256 * 256, 0);

    // Original MATLAB:
    // for i=1:256
    //   for j=1:256
    //     dist=inf;
    //     for k=1:256
    //       if abs(B(i,j)-E_C(k))<=dist && N1(k)~=0
    //          ... index(i,j)=k;
    //       end
    //     end
    //   end
    // end
    for (mwSize j = 0; j < 256; ++j) {
        for (mwSize i = 0; i < 256; ++i) {
            double Bij = E_C[i] * E_C[j];
            double dist = std::numeric_limits<double>::infinity();
            int best = 0;
            for (mwSize k = 0; k < 256; ++k) {
                double d = std::abs(Bij - E_C[k]);
                if (d <= dist && N1[k] != 0.0) {
                    dist = d;
                    best = static_cast<int>(k);
                }
            }
            index[IDX2(i,j)] = best;
        }
    }

    // weight_matrix(i,j)=1/(sum(sum(index==index(i,j)))+eps);
    // Use counts of each index value to avoid repeatedly scanning the full matrix.
    std::vector<int> idxCount(256, 0);
    for (mwSize t = 0; t < index.size(); ++t) {
        idxCount[index[t]]++;
    }
    const double epsv = mxGetEps();
    std::vector<double> weight(256 * 256, 0.0);
    for (mwSize j = 0; j < 256; ++j) {
        for (mwSize i = 0; i < 256; ++i) {
            int id = index[IDX2(i,j)];
            weight[IDX2(i,j)] = 1.0 / (static_cast<double>(idxCount[id]) + epsv);
        }
    }

    std::vector<double> L(256), L_new(256, 0.0), R(256, 0.0), R_old(256, 0.0);
    for (mwSize i = 0; i < 256; ++i) L[i] = L_init[i];

    int iter = 1;
    while (true) {
        if (iter != 1) {
            for (mwSize i = 0; i < 256; ++i) {
                L[i] = L_new[i];
                R_old[i] = R[i];
            }
        }

        // R update
        double sumL2 = 0.0;
        for (mwSize i = 0; i < 256; ++i) sumL2 += L[i] * L[i];
        double de = alpha + sumL2 / (N * N);

        for (mwSize i = 0; i < 256; ++i) {
            double mo = 0.0;
            for (mwSize j = 0; j < 256; ++j) {
                int id = index[IDX2(i,j)];
                mo += weight[IDX2(i,j)] * L[j] * N1[id];
            }
            mo /= N;
            R[i] = (mo + alpha * L6[i]) / de;
        }

        // L update
        double sumR2 = 0.0;
        for (mwSize i = 0; i < 256; ++i) sumR2 += R[i] * R[i];
        de = beta + sumR2 / (N * N);

        for (mwSize i = 0; i < 256; ++i) {
            double mo = 0.0;
            for (mwSize j = 0; j < 256; ++j) {
                int id = index[IDX2(j,i)]; // index(:,i) in MATLAB
                mo += weight[IDX2(j,i)] * R[j] * N1[id];
            }
            mo /= N;
            L_new[i] = (mo + beta * L5[i]) / de;
        }

        // S1 = R .* L_new' / N; new_hist accumulation
        std::vector<double> new_hist(256, 0.0);
        for (mwSize j = 0; j < 256; ++j) {
            for (mwSize i = 0; i < 256; ++i) {
                double s = R[i] * L_new[j] / N;
                int id = index[IDX2(i,j)];
                new_hist[id] += s;
            }
        }

        // weight_matrix update: weight(i,j)=S1(i,j)/(new_hist(index(i,j))+eps)
        for (mwSize j = 0; j < 256; ++j) {
            for (mwSize i = 0; i < 256; ++i) {
                double s = R[i] * L_new[j] / N;
                int id = index[IDX2(i,j)];
                weight[IDX2(i,j)] = s / (new_hist[id] + epsv);
            }
        }

        double diffL = 0.0, diffR = 0.0;
        for (mwSize i = 0; i < 256; ++i) {
            diffL += std::abs(L_new[i] - L[i]);
            diffR += std::abs(R_old[i] - R[i]);
        }

        // Original: if sum(abs(L_new-L))<0.1 && sum(abs(R_old-R))<0.1 || iter>3000
        // External interface: replace 3000 with maxIter, keep original condition form.
        if ((diffL < tolL && diffR < tolR) || iter > maxIter) {
            break;
        }
        iter++;
    }

    // Normalize L_new and R
    double sumLnew = 0.0, sumR = 0.0;
    for (mwSize i = 0; i < 256; ++i) {
        sumLnew += L_new[i];
        sumR += R[i];
    }
    if (sumLnew != 0.0) {
        for (mwSize i = 0; i < 256; ++i) L_new[i] = L_new[i] * N / sumLnew;
    }
    if (sumR != 0.0) {
        for (mwSize i = 0; i < 256; ++i) R[i] = R[i] * N / sumR;
    }

    // Contrast stretching mapping
    std::vector<double> C2(256);
    for (mwSize i = 0; i < 256; ++i) C2[i] = std::pow(C1[i], 1.0 / gamma);

    std::vector<int> ind(256, 0);
    for (mwSize i = 0; i < 256; ++i) {
        double dist = std::numeric_limits<double>::infinity();
        int best = 0;
        for (mwSize j = 0; j < 256; ++j) {
            double d = std::abs(C2[i] - C1[j]);
            // Original uses <, not <=
            if (d < dist) {
                dist = d;
                best = static_cast<int>(j);
            }
        }
        ind[i] = best;
    }

    std::vector<double> new_L(256, 0.0);
    for (mwSize i = 0; i < 256; ++i) {
        new_L[ind[i]] += L_new[i];
    }

    // index_2 according to C3 = C1*C1'
    std::vector<int> index2(256 * 256, 0);
    for (mwSize j = 0; j < 256; ++j) {
        for (mwSize i = 0; i < 256; ++i) {
            double C3ij = C1[i] * C1[j];
            double dist = std::numeric_limits<double>::infinity();
            int best = 0;
            for (mwSize k = 0; k < 256; ++k) {
                double d = std::abs(C3ij - C1[k]);
                // Original uses <=
                if (d <= dist) {
                    dist = d;
                    best = static_cast<int>(k);
                }
            }
            index2[IDX2(i,j)] = best;
        }
    }

    std::vector<double> result2(256, 0.0);
    for (mwSize j = 0; j < 256; ++j) {
        for (mwSize i = 0; i < 256; ++i) {
            double Aij = R[i] * new_L[j];
            int id = index2[IDX2(i,j)];
            result2[id] += Aij;
        }
    }
    for (mwSize i = 0; i < 256; ++i) result2[i] /= N;

    // Outputs
    plhs[0] = mxCreateDoubleMatrix(256, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(256, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(256, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(256, 1, mxREAL);

    double* outLnew = mxGetPr(plhs[0]);
    double* outNewL = mxGetPr(plhs[1]);
    double* outR = mxGetPr(plhs[2]);
    double* outResult = mxGetPr(plhs[3]);

    for (mwSize i = 0; i < 256; ++i) {
        outLnew[i] = L_new[i];
        outNewL[i] = new_L[i];
        outR[i] = R[i];
        outResult[i] = result2[i];
    }

    if (nlhs == 5) {
        plhs[4] = mxCreateDoubleScalar(static_cast<double>(iter));
    }
}
