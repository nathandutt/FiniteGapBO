#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <iostream>
#include <utility>
#include <cassert>
#include <iomanip>
#define FIXED_FLOAT(x) std::fixed <<std::setprecision(1)<<(x)
#include <Eigen/Dense>

using namespace std;
using Matrix = Eigen::MatrixXcd;
using Vector  = Eigen::VectorXcd;
using cplx = std::complex<double>;
const cplx I(0., 1.);


//---------------Functions to reconstruct the field from poisson kernels------//

double poissonKernel(double x, const cplx& q){
    //Expression for a single Poisson kernel of complex parameter q
    //The phase of q is a shift in the x coord
    double r = abs(q); double x0 = -arg(q);
    double x_shifted = x-x0;
    return (1-r*r)/(1+r*r-2.*r*cos(x_shifted));
}

double pointField(double x, const Vector& qs){
    //construct sum of Poisson kernels corresponding to all qs
    //at a specific point x
    int N = qs.size();
    double field = 0.;
    for(int i = 0; i < N; i++){
	field += poissonKernel(x, qs(i));
    }
    return field;
}

vector<double> fullField(const vector<double>& xs, const Vector& qs){
    //construct full field of Poisson kernels over a given lattice of xs
    int s = xs.size();
    vector<double> field{};
    field.reserve(s);
    for(int i = 0; i < s; i++){
	field.emplace_back(pointField(xs[i], qs));
    }
    return field;
}

//---------------------------------------------------------------------------//
vector<cplx> calculateCs(const vector<double>& ls){

    //Helper function to caluclate certain coefficients in Krichever's multiphase solution
    int N = ls.size();
    vector<cplx> cs{};
    cs.reserve(N);
    for(int i =0; i < N; i++){
	cs.emplace_back(cplx(-1., 0.));
	for(int j = 0; j < N; j++){
	    cs[i]*= 1./(ls[i] + 1 - ls[j])/(ls[i]-ls[j]-1);
	    if(j==i) {cs[i]*=(ls[i]+1)/ls[i]; continue;}
	    cs[i] *= (ls[i] - ls[j])*(ls[i]-ls[j]);
	}
	cs[i] = sqrt(cs[i]);
    }
    return cs;
}

Vector diagonalize(const Matrix& m, double& cumulative_err){
    //Diagonalize a matrix
    //Keep track of maximal inversion and diagonalization error
    int N = m.rows();
    Eigen::ComplexEigenSolver<Matrix> solver(m);
    Vector eigenvalues_vec = solver.eigenvalues();

    Matrix VR              = solver.eigenvectors();
    Matrix VL = VR.inverse();

    Matrix Lambda = Matrix::Zero(N, N);
    for(int i = 0; i < N; i++) Lambda(i,i) = eigenvalues_vec(i);
    double diag_err = (VR * Lambda * VL - m).norm();
    double inv_err = (VL * VR - Matrix::Identity(N,N)).norm();
    double err = max(diag_err, inv_err);
    cumulative_err = max(err, cumulative_err);

    return eigenvalues_vec;
}

Matrix fromSpectrum(const vector<cplx>& cs, const vector<double>& ls, const vector<double>& phases, double t){
    //Construct matrix from scattering data
    //The eigenvalues for a specific time will correpsond to the poisson kernels making up the field
    assert((ls.size() == phases.size()));
    int N = ls.size();
    Matrix m(N,N);
    for(int i = 0; i < N; i++){
	m(i, i) = 1./cs[i] * exp(-I*phases[i]-I*(2*ls[i]+1)*t);
	for(int j=0; j<N; j++){
	    if(i==j) continue;
	    m(i,j) = 1./cs[i] * exp(-I*phases[i]-I*(2*ls[i]+1)*t)/(ls[i] - ls[j] + 1.);
	}
    }
    return m;
}

Matrix laxFromQs(const vector<cplx>& qs){
    //Construct lax matrix from qs
    int N = qs.size();
    Matrix l(N,N);
    for(int i = 0; i < N; i++){
	l(i,i) = cplx(0., 0.);
	for(int j=0; j < N; j++){
	    l(i,i) += -1./(1.-conj(qs[j])*qs[i]);
	    if(i==j) continue;
	    l(i,i) += +qs[j]/(qs[j]-qs[i]);
	    l(i,j) = -qs[i]/(qs[i]-qs[j]);
	}
    }
    return l;
}

pair<Vector, Matrix> diagonalizeLax(const Matrix& lax, const vector<cplx>& qs, double& cum_err){
    int N = lax.rows();
    Eigen::ComplexEigenSolver<Matrix> solver(lax);

    //Diagonalize the lax matrix first, and track diagonalization error.
    Vector lax_eigs = solver.eigenvalues();

    Matrix VR              = solver.eigenvectors();
    Matrix VL = VR.inverse();

    Matrix Lambda = Matrix::Zero(N, N);
    for(int i = 0; i < N; i++) Lambda(i,i) = lax_eigs(i);
    double diag_err = (VR * Lambda * VL - lax).norm();
    double inv_err = (VL * VR - Matrix::Identity(N,N)).norm();
    double err = max(diag_err, inv_err);
    cum_err = max(err, cum_err);


    //Define Q
    Matrix Q = Matrix::Zero(N, N);
    for(int i = 0; i < N; i++) Q(i, i) = qs[i];

    // Now calculate Q in eigenvector basis
    Matrix basis_change = VL*Q*VR;

    return {lax_eigs, basis_change};

}
Vector evolvedQs(const Vector& lax_eigs, const Matrix& Qrotated, double t, double& cum_err){

    int N = lax_eigs.size();
    Matrix expL = Matrix::Zero(N, N);
    for(int i = 0; i < N; i++) expL(i, i) = exp(2.*I*t*lax_eigs(i) + I*t);

    Matrix full = expL*Qrotated;
    Vector eigs = diagonalize(full, cum_err);
    return eigs;
}

vector<vector<double>> localIST(const vector<vector<cplx>>& binned_qs){
    //Assume that binned_qs have been rescaled to bin already
    int num_bins = binned_qs.size();

    vector<vector<double>> result{};
    result.reserve(num_bins);

    for(int bin_idx = 0; bin_idx < num_bins; bin_idx++){

	int local_phase_num = binned_qs[bin_idx].size();
	vector<double> bin_result{};
	bin_result.reserve(local_phase_num);
	if (local_phase_num > 0){

	    //Diagonalize local bin lax matrix
	    Matrix lax_bin = laxFromQs(binned_qs[bin_idx]);
	    Eigen::ComplexEigenSolver<Matrix> solver(lax_bin);
	    Vector eigs = solver.eigenvalues();
	    //No error checking here as bins will usually have smaller amount of phases

	    for(int i = 0; i < local_phase_num; i++){
		bin_result.emplace_back(eigs(i).real());
	    }
	}
	result.emplace_back(bin_result);
    }
    return result;
}

vector<Vector> inverseScatter(vector<double> ls, vector<double> phases, 
	vector<double> times)
{
    /*
     * Calculates the multiphase solution at all given times over the lattice of xs
     * using inverse scattering given the spectrum in "ls", and norming constants in "phases"
     */

    //initialize return vector
    vector<Vector> all_qs{}; all_qs.reserve(times.size());

    //calculate finitegap coefficients
    vector<cplx> cs = calculateCs(ls);

    double cum_err = 0.;

    int num_times = times.size();
    int i = 0;
    for(const auto t : times){
	double percent = (double)(100*i)/num_times;
	i++;

	cout << "\r\033[K" << "---- " << FIXED_FLOAT(percent) << " % Done ----" << std::flush;
	Matrix m_t = fromSpectrum(cs, ls, phases, t);
	Vector qs_t = diagonalize(m_t, cum_err);
	all_qs.emplace_back(qs_t);	
    }

    cout  << std::defaultfloat<< "\n" << "Maximal inversion/diagonalization error encountered is: " << cum_err << "\n \n";
    return all_qs;
}

vector<Vector> directEvolve(vector<cplx> qs, vector<double> times){

    int N = qs.size();

    vector<Vector> all_qs{}; all_qs.reserve(times.size());

    Matrix l = laxFromQs(qs);
    double cum_err = 0.;
    auto [lax_eigs, Qrotated] = diagonalizeLax(l, qs, cum_err);

    int num_times = times.size();
    int i = 0;
    for(const auto t : times){
	double percent = (double)(100*i)/num_times;
	i++;

	cout << "\r\033[K" << "---- " << FIXED_FLOAT(percent) << " % Done ----" << std::flush;
	Vector qs_t = evolvedQs(lax_eigs, Qrotated, t, cum_err);
	all_qs.emplace_back(qs_t);
    }

    cout  << std::defaultfloat << "\n" << "Maximal inversion/diagonalization error encountered is: " << cum_err << "\n";
    return all_qs;
}




//Bind inverseScatter to a python module
PYBIND11_MODULE(sim, m){
    m.def("inverseScatter", &inverseScatter, "calculate field from scattering data");
    m.def("directEvolve", &directEvolve, "calculate field from intial poisson kernels");
    m.def("localIST", &localIST, "calculate local lax eigenvals from binned qs");
}
