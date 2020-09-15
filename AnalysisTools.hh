// functions and methods for jackknife analysis and multivariable function fits for correlated data sets

// used headers/libraries
#include <Eigen/Dense>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_bessel.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <numeric>

// ------------------------------------------------------------------------------------------------------------

//
//
// READING GIVEN DATASET FOR FURTHER ANALYSIS
//
//

// read file with dataset into a raw matrix
Eigen::MatrixXd ReadFile(std::string const &fileName)
{
    // start reading
    std::ifstream fileToRead;
    fileToRead.open(fileName);

    // determine number of columns
    std::string firstLine;
    std::getline(fileToRead, firstLine);
    std::stringstream firstLineStream(firstLine);

    // number of columns in given file
    int numOfCols = 0;
    std::string tmpString;
    // count number of writes to a temporary string container
    while (firstLineStream >> tmpString)
    {
        numOfCols++;
    }
    fileToRead.close();

    // string for all the lines
    std::string line;

    // data structure (raw matrix) to store data
    Eigen::MatrixXd rawDataMat(0, numOfCols);

    // reopen file
    fileToRead.open(fileName);
    // check if open
    if (fileToRead.is_open())
    {
        // read line by line
        int i = 0;
        while (std::getline(fileToRead, line))
        {
            // using stringstream to write matrix
            std::stringstream dataStream(line);
            rawDataMat.conservativeResize(i + 1, numOfCols);
            for (int j = 0; j < numOfCols; j++)
            {
                dataStream >> rawDataMat(i, j);
            }
            i++;
        }
        // close file
        fileToRead.close();
    }
    // error check
    else
    {
        std::cout << "ERROR\nProblem occured while reading given file." << std::endl;
        std::exit(-1);
    }

    // return raw data matrix
    return rawDataMat;
}

//
//
// CALCULATING SUSCEPTIBILITIES (with jackknife samples --> vector form)
// labeling
// imZu --> ZContainer[0];
// imZs --> ZContainer[1];
// Zuu  --> ZContainer[2];
// Zud  --> ZContainer[3];
// Zus  --> ZContainer[4];
// Zss  --> ZContainer[5];
//
//

// imZB
auto imZBCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return (2 * Z[0] + Z[1]) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// imZQ
auto imZQCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return (Z[0] - Z[1]) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// imZS
auto imZSCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return -Z[1];
};

// ------------------------------------------------------------------------------------------------------------

// ZBB
auto ZBBCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return (2 * Z[2] + Z[5] + 4 * Z[4] + 2 * Z[3]) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZQQ
auto ZQQCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return (5 * Z[2] + Z[5] - 2 * Z[4] - 4 * Z[3]) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZSS
auto ZSSCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return Z[5];
};

// ------------------------------------------------------------------------------------------------------------

// ZII
auto ZIICalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return (Z[2] - Z[3]) / 2;
};

// ------------------------------------------------------------------------------------------------------------

// ZBQ
auto ZBQCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return (Z[2] - Z[5] - Z[4] + Z[3]) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZBS
auto ZBSCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return -(Z[5] + 2 * Z[4]) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// ZQS
auto ZQSCalc = [](std::vector<Eigen::VectorXd> const &Z) {
    return (Z[5] - Z[4]) / 3;
};

//
//
// STATISTICAL FUNCTIONS (ERROS, VARIANCE, JACKKNIFE, ETC...)
// including sample number reduction methods
//
//

// calculate variance (for jackknife samples)
auto JCKVariance = [](Eigen::VectorXd const &JCKSamples) {
    // size of vector
    int N = JCKSamples.size();
    // pre-factor
    double preFactor = (double)(N - 1) / N;
    // estimator / mean
    double estimator = JCKSamples.mean();
    // calculate variance
    double var = 0.;
    for (int i = 0; i < N; i++)
    {
        double val = JCKSamples(i) - estimator;
        var += val * val;
    }
    // return variance
    return preFactor * var;
};

// ------------------------------------------------------------------------------------------------------------

// general jackknife error calculator for susceptibilities
auto ZError = [](Eigen::VectorXd const &Z) {
    return std::sqrt(JCKVariance(Z.segment(2, Z.size() - 2)));
};

// ------------------------------------------------------------------------------------------------------------

// calculate original block means (and reducing their number by averaging) from jackknife samples
auto JCKReducedBlocks = [](Eigen::VectorXd const &JCKSamplesOld, int const &divisor) {
    // number of samples
    int const NOld = JCKSamplesOld.size();
    // test if divisor is correct for the original sample number
    if ((NOld % divisor) != 0.)
    {
        std::cout << "ERROR\nIncorrect divisor during Jackknife sample reduction." << std::endl;
        std::exit(-1);
    }
    // empty vector for block values
    Eigen::VectorXd blockVals(NOld);
    // sum of (original) samples
    double const sum = JCKSamplesOld.sum();
    // calculate block values and add to vector
    for (int i = 0; i < NOld; i++)
    {
        blockVals(i) = sum - (NOld - 1) * JCKSamplesOld(i);
    }
    // create new blocks
    // old blocks to add up for new blocks
    int const reduced = NOld / divisor;
    // vector for new blocks (reduced)
    Eigen::VectorXd newBlocks(divisor);
    // calculate new blocks
    for (int i = 0; i < divisor; i++)
    {
        newBlocks(i) = 0;
        for (int j = 0; j < reduced; j++)
        {
            newBlocks(i) += blockVals(i * reduced + j);
        }
        newBlocks(i) /= reduced;
    }
    // return new blocks
    return newBlocks;
};

// ------------------------------------------------------------------------------------------------------------

// calculate jackknife samples from block means
auto JCKSamplesCalculation = [](Eigen::VectorXd const &Blocks) {
    // number of blocks
    int const lengthBlocks = Blocks.size();
    // vector for jackknife samples
    Eigen::VectorXd Samples(lengthBlocks);
    // copy data to std::vector
    std::vector<double> tempVec(Blocks.data(), Blocks.data() + lengthBlocks);
    // create jackknife samples
    for (int i = 0; i < lengthBlocks; i++)
    {
        // copy data
        std::vector<double> tempJCKVec = tempVec;
        // delete ith element
        tempJCKVec.erase(tempJCKVec.begin() + i);
        // calculate mean
        Samples[i] = std::accumulate(tempJCKVec.begin(), tempJCKVec.end(), 0.) / (lengthBlocks - 1);
    }
    // return new jackknife samples
    return Samples;
};

// ------------------------------------------------------------------------------------------------------------

// general jackknife error calculator for susceptibilities with sample number reductions (according to divisors)
auto ZErrorJCKReduced = [](Eigen::VectorXd const &Z, int const &divisor) {
    // number of jackknife samples
    int NOld = Z.size() - 2;
    // get new jackknife samples via calculating old blocks and reducing their number by averaging
    Eigen::VectorXd JCKSamples = JCKSamplesCalculation(JCKReducedBlocks(Z.segment(2, NOld), divisor));
    // return jackknfife error
    return std::sqrt(JCKVariance(JCKSamples));
};

//
//
// FUNCTION FITTING METHODS (2D and/or correlated)
//
//

// calculate correlation coefficients of two datasets with given means (better this way)
auto CorrCoeff = [](Eigen::VectorXd const &vec1, Eigen::VectorXd const &vec2, double const &mean1, double const &mean2) {
    // number of jackknife samples
    double NJck = (double)vec1.size();

    // calculate correlation (not normed)
    double corr = 0;
    for (int i = 0; i < NJck; i++)
    {
        corr += (vec1(i) - mean1) * (vec2(i) - mean2);
    }

    // return normed correlation coefficient
    return corr * (NJck - 1) / NJck;
};

// ------------------------------------------------------------------------------------------------------------

// block from the blockdiagonal covariance matrix
auto BlockCInverse = [](Eigen::MatrixXd const &dataToFitMat, int const &numOfQs, int const &qIndex, int const &jckNum) {
    // jackknife samples to calculate/estimate correlations
    Eigen::MatrixXd JCKs(numOfQs, jckNum);
    for (int i = 0; i < JCKs.cols(); i++)
    {
        for (int j = 0; j < JCKs.rows(); j++)
        {
            JCKs(j, i) = dataToFitMat(qIndex * numOfQs + j, i + 1);
        }
    }

    // get y_err data to compare with correlation results (maybe later)

    // means to calculate correlations
    std::vector<double> means(numOfQs, 0.);
    for (int i = 0; i < numOfQs; i++)
    {
        means[i] = JCKs.row(i).mean();
    }

    // covariance matrix block
    Eigen::MatrixXd C(numOfQs, numOfQs);
    for (int i = 0; i < numOfQs; i++)
    {
        for (int j = i; j < numOfQs; j++)
        {
            // triangular part
            C(j, i) = CorrCoeff(JCKs.row(i), JCKs.row(j), means[i], means[j]);
            std::cout << C(j, i) << std::endl;
            // using symmetries
            if (i != j)
                C(i, j) = C(j, i);
        }
    }

    // return inverse covariance matrix block
    return (Eigen::MatrixXd)C.inverse();
};

// ------------------------------------------------------------------------------------------------------------



//
//
// HADRON RESONANCE GAS (HRG) FUNCTIONS
//
//

// lambda to calculate squares
auto sq = [](auto const &x) {
    return x * x;
};

// ------------------------------------------------------------------------------------------------------------

// determine eta function for given particle type (boson / fermion)
auto EtaDetermination = [](std::string const &particleType) {
    int eta = 0;
    if (particleType == "boson")
        eta = -1;
    else if (particleType == "fermion")
        eta = 1;
    else
    {
        std::cout << "ERROR\nGiven particle type is not appropriate." << std::endl;
        std::exit(-1);
    }

    // return appropriate eta-value
    return eta;
};

// ------------------------------------------------------------------------------------------------------------

// partial pressure calculator (for dimension = 3) at mu = 0
// kCut index is not included in the final summation
auto iPartialPressure = [](double const &temperature, int const &iSpinDeg, double const &iHadronMass, std::string const &particleType, int const &kCut) {
    // determine particle type (boson / fermion)
    int eta = EtaDetermination(particleType);

    // pre-factor
    double preFactor = iSpinDeg * sq(temperature * iHadronMass / M_PI) / 2;

    // summation of Macdonald function
    double sumBessel = 0.;
    for (int k = 1; k < kCut; k++)
    {
        double argumentBessel = k * iHadronMass / temperature;
        sumBessel += std::pow(-eta, k + 1) / sq(k) * gsl_sf_bessel_Kn(2, argumentBessel);
    }

    // return partial pressure
    return preFactor * sumBessel;
};

// ------------------------------------------------------------------------------------------------------------

// partial energy density calculator (for dimension = 3) at mu = 0
// kCut index is not included in the final summation
auto iPartialEnergyDensity = [](double const &temperature, int const &iSpinDeg, double const &iHadronMass, std::string const &particleType, int const &kCut) {
    // determine particle type (boson / fermion)
    int eta = EtaDetermination(particleType);

    // pre-factor
    double preFactor = iSpinDeg * sq(temperature * iHadronMass / M_PI) / 2;

    // summation of Macdonald function
    double sumBessel = 0.;
    for (int k = 1; k < kCut; k++)
    {
        double argumentBessel = k * iHadronMass / temperature;
        sumBessel += std::pow(-eta, k + 1) / sq(k) * (3 * gsl_sf_bessel_Kn(2, argumentBessel) + argumentBessel * gsl_sf_bessel_Kn(1, argumentBessel));
    }

    // return partial energy density
    return preFactor * sumBessel;
};

// ------------------------------------------------------------------------------------------------------------

// partial trace anomaly (interaction measure; for dimension = 3) at mu = 0
auto iPartialTraceAnomaly = [](double const &partialPressure, double const &partialEnergyDensity) {
    // return trace anomaly
    return partialEnergyDensity - 3 * partialPressure;
};

// ------------------------------------------------------------------------------------------------------------

// partial (even) suscebtibility calculator (for dimension = 3) at mu = 3 (pressure and chemical potentials are reduced)
// kCut index is not included in the final summation
auto iPartialSusceptibility = [](int const &orderB, int const &orderS, int const &orderQ, double const &temperature, int const &iSpinDeg, double const &iHadronMass, std::string const &particleType, int const &kCut, int const &iBaryonNumber, int const &iStrangeness, int const &iElectricCharge) {
    // check if orders are even
    if ((orderB + orderS + orderQ) % 2 != 0)
    {
        std::cout << "ERROR\nGiven susceptibility orders are not appropriate." << std::endl;
        std::exit(-1);
    }

    // determine particle type (boson / fermion)
    int eta = EtaDetermination(particleType);

    // pre-factor
    double preFactor = iSpinDeg * sq(iHadronMass / M_PI / temperature) / 2;

    // summation of Macdonald function
    double sumBessel = 0.;
    for (int k = 1; k < kCut; k++)
    {
        double argumentBessel = k * iHadronMass / temperature;
        sumBessel += std::pow(-eta, k + 1) / sq(k) * std::pow(k * iBaryonNumber, orderB) * std::pow(k * iStrangeness, orderS) * std::pow(k * iElectricCharge, orderQ) * gsl_sf_bessel_Kn(2, argumentBessel);
    }

    // return partial susceptibility
    return preFactor * sumBessel;
};
