#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <fstream>
#include <sstream>
#include <iostream>
#include <Eigen/Dense>
#include <cassert>

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

vector<vector<double>> inverseScatter(vector<double> ls, vector<double> phases, 
							    vector<double> times, vector<double> xs)
{
    /*
     * Calculates the multiphase solution at all given times over the lattice of xs
     * using inverse scattering given the spectrum in "ls", and norming constants in "phases"
     */
    
    //initialize return vector
    vector<vector<double>> all_fields{}; all_fields.reserve(times.size());

    //calculate finitegap coefficients
    vector<cplx> cs = calculateCs(ls);

    double cum_err = 0.;

    for(const auto t : times){
	cout << "\r\033[K" << "T= " << t << std::flush;
	Matrix m_t = fromSpectrum(cs, ls, phases, t);
	Vector qs_t = diagonalize(m_t, cum_err);
	vector<double> field_t = fullField(xs, qs_t);
	all_fields.emplace_back(field_t);
    }

    cout << "Maximal inversion/diagonalization error encountered is: " << cum_err << "\n";
    return all_fields;
}

vector<vector<double>> directEvolve(vector<double> rs, vector<double> shifts, vector<double> times, vector<double> xs){
    /*
     * Evolves the poisson kernels directly, no scattering data
     */
    assert((rs.size() == shifts.size()));
    int N = rs.size();

    vector<vector<double>> all_fields{}; all_fields.reserve(times.size());

    //Create qs
    vector<cplx> qs{}; qs.reserve(N);
    for(int i=0; i<N; i++){
	qs.emplace_back(rs[i]*exp(-I*shifts[i]));
    }

    Matrix l = laxFromQs(qs);
    double cum_err = 0.;
    auto [lax_eigs, Qrotated] = diagonalizeLax(l, qs, cum_err);

    for(const auto t : times){
	cout << "\r\033[K" << "T= " << t << std::flush;
	Vector qs_t = evolvedQs(lax_eigs, Qrotated, t, cum_err);
	vector<double> field_t = fullField(xs, qs_t);
	all_fields.emplace_back(field_t);
    }

    cout << "Maximal inversion/diagonalization error encountered is: " << cum_err << "\n";
    return all_fields;
}




//Bind inverseScatter to a python module
PYBIND11_MODULE(sim, m){
    m.def("inverseScatter", &inverseScatter, "calculate field from scattering data");
    m.def("directEvolve", &directEvolve, "calculate field from intial poisson kernels");
}
