// including used header
#include "AnalysisTools.hh"

// ------------------------------------------------------------------------------------------------------------

// PDF file name (hadron list)
std::string const PDG = "../PDG.txt";

// ------------------------------------------------------------------------------------------------------------

// main function
// argv[1] --> name of given file with dataset
// argv[2] --> number of jackknife samples
// argv[3] --> number of used susceptibilities
int main(int argc, char **argv)
{
    // check argument list
    if (argc < 4)
    {
        std::cout << "ERROR\nNot enough arguments given." << std::endl;
        std::exit(-1);
    }

    // prepare for reading given file
    // string for file name
    std::string const fileName = argv[1];
    // matrix container for raw data for analysis
    Eigen::MatrixXd const rawDataMat = ReadFile(fileName);

    // number of jackknife samples
    int const jckNum = std::atoi(argv[2]);
    // number of used susceptibilities
    int const ZNum = std::atoi(argv[3]);

    // number of cols and rows of raw data matrix
    //int const cols = rawDataMat.cols();
    int const rows = rawDataMat.rows();

    // chemical potentials for baryon numbers and strangeness
    Eigen::VectorXd const muB = rawDataMat.col(2);
    Eigen::VectorXd const muS = rawDataMat.col(3);

    // susceptibilities (regarding u, d, s flavours) with error and jackknife samples
    // size of vectors
    int const ZSize = 2 + jckNum;
    // vectors to store imZB values and their estimated errors (results) + JCK
    Eigen::VectorXd imZBVals(rows);
    Eigen::VectorXd imZBErrs(rows);
    Eigen::MatrixXd imZBJCKs(rows, jckNum);
    // vectors to store imZQ values and their estimated errors (results) + JCK
    Eigen::VectorXd imZQVals(rows);
    Eigen::VectorXd imZQErrs(rows);
    Eigen::MatrixXd imZQJCKs(rows, jckNum);
    // vectors to store imZS values and their estimated errors (results) + JCK
    Eigen::VectorXd imZSVals(rows);
    Eigen::VectorXd imZSErrs(rows);
    Eigen::MatrixXd imZSJCKs(rows, jckNum);
    // vectors to store ZBB values and their estimated errors (results) + JCK
    Eigen::VectorXd ZBBVals(rows);
    Eigen::VectorXd ZBBErrs(rows);
    Eigen::MatrixXd ZBBJCKs(rows, jckNum);
    // vectors to store ZQQ values and their estimated errors (results) + JCK
    Eigen::VectorXd ZQQVals(rows);
    Eigen::VectorXd ZQQErrs(rows);
    Eigen::MatrixXd ZQQJCKs(rows, jckNum);
    // vectors to store ZSS values and their estimated errors (results) + JCK
    Eigen::VectorXd ZSSVals(rows);
    Eigen::VectorXd ZSSErrs(rows);
    Eigen::MatrixXd ZSSJCKs(rows, jckNum);
    // vectors to store ZBQ values and their estimated errors (results) + JCK
    Eigen::VectorXd ZBQVals(rows);
    Eigen::VectorXd ZBQErrs(rows);
    Eigen::MatrixXd ZBQJCKs(rows, jckNum);
    // vectors to store ZBS values and their estimated errors (results) + JCK
    Eigen::VectorXd ZBSVals(rows);
    Eigen::VectorXd ZBSErrs(rows);
    Eigen::MatrixXd ZBSJCKs(rows, jckNum);
    // vectors to store ZQS values and their estimated errors (results) + JCK
    Eigen::VectorXd ZQSVals(rows);
    Eigen::VectorXd ZQSErrs(rows);
    Eigen::MatrixXd ZQSJCKs(rows, jckNum);
    // vectors to store ZII values and their estimated errors (results) + JCK
    Eigen::VectorXd ZIIVals(rows);
    Eigen::VectorXd ZIIErrs(rows);
    Eigen::MatrixXd ZIIJCKs(rows, jckNum);
    // container for "flavour vectors"
    std::vector<Eigen::VectorXd> ZContainer(ZNum);
    // loop for every row
    for (int i = 0; i < rows; i++)
    {
        // filling up vectors
        for (int j = 0; j < ZNum; j++)
        {
            ZContainer[j] = rawDataMat.row(i).segment(4 + j * ZSize, ZSize);
        }
        // calculate (in vector form with jackknife samples)
        Eigen::VectorXd imZB = imZBCalc(ZContainer);
        Eigen::VectorXd imZQ = imZQCalc(ZContainer);
        Eigen::VectorXd imZS = imZSCalc(ZContainer);
        Eigen::VectorXd ZBB = ZBBCalc(ZContainer);
        Eigen::VectorXd ZQQ = ZQQCalc(ZContainer);
        Eigen::VectorXd ZSS = ZSSCalc(ZContainer);
        Eigen::VectorXd ZBQ = ZBQCalc(ZContainer);
        Eigen::VectorXd ZBS = ZBSCalc(ZContainer);
        Eigen::VectorXd ZQS = ZQSCalc(ZContainer);
        Eigen::VectorXd ZII = ZIICalc(ZContainer);
        // save results
        // imZB
        imZBVals(i) = imZB(0);
        // imZQ
        imZQVals(i) = imZQ(0);
        // imZS
        imZSVals(i) = imZS(0);
        // ZBB
        ZBBVals(i) = ZBB(0);
        // ZQQ
        ZQQVals(i) = ZQQ(0);
        // ZSS
        ZSSVals(i) = ZSS(0);
        // ZBQ
        ZBQVals(i) = ZBQ(0);
        // ZBS
        ZBSVals(i) = ZBS(0);
        // ZQS
        ZQSVals(i) = ZQS(0);
        // ZII
        ZIIVals(i) = ZII(0);

        // save jackknife samples
        // imZB
        imZBJCKs.row(i) = imZB.segment(2, jckNum);
        // imZQ
        imZQJCKs.row(i) = imZQ.segment(2, jckNum);
        // imZS
        imZSJCKs.row(i) = imZS.segment(2, jckNum);
        // ZBB
        ZBBJCKs.row(i) = ZBB.segment(2, jckNum);
        // ZQQ
        ZQQJCKs.row(i) = ZQQ.segment(2, jckNum);
        // ZSS
        ZSSJCKs.row(i) = ZSS.segment(2, jckNum);
        // ZBQ
        ZBQJCKs.row(i) = ZBQ.segment(2, jckNum);
        // ZBS
        ZBSJCKs.row(i) = ZBS.segment(2, jckNum);
        // ZQS
        ZQSJCKs.row(i) = ZQS.segment(2, jckNum);
        // ZII
        ZIIJCKs.row(i) = ZII.segment(2, jckNum);

        // decide error estimation method
        // errors with jacknife sample number reduction OFF
        if (argc < 6)
        {
            // errors
            imZBErrs(i) = ZError(imZB);
            imZQErrs(i) = ZError(imZQ);
            imZSErrs(i) = ZError(imZS);
            ZBBErrs(i) = ZError(ZBB);
            ZQQErrs(i) = ZError(ZQQ);
            ZSSErrs(i) = ZError(ZSS);
            ZBQErrs(i) = ZError(ZBQ);
            ZBSErrs(i) = ZError(ZBS);
            ZQSErrs(i) = ZError(ZQS);
            ZIIErrs(i) = ZError(ZII);
        }
        // errors with jacknife sample number reduction ON
        else if (argc == 6)
        {
            // divisor (number of new samples)
            int const divisor = std::atoi(argv[5]);
            // errors
            imZBErrs(i) = ZErrorJCKReduced(imZB, divisor);
            imZQErrs(i) = ZErrorJCKReduced(imZQ, divisor);
            imZSErrs(i) = ZErrorJCKReduced(imZS, divisor);
            ZBBErrs(i) = ZErrorJCKReduced(ZBB, divisor);
            ZQQErrs(i) = ZErrorJCKReduced(ZQQ, divisor);
            ZSSErrs(i) = ZErrorJCKReduced(ZSS, divisor);
            ZBQErrs(i) = ZErrorJCKReduced(ZBQ, divisor);
            ZBSErrs(i) = ZErrorJCKReduced(ZBS, divisor);
            ZQSErrs(i) = ZErrorJCKReduced(ZQS, divisor);
            ZIIErrs(i) = ZErrorJCKReduced(ZII, divisor);
        }
    }

    //
    // START FIT
    //

    // number of x-values (muB and muS)
    int const N = muB.size();

    // number of quantities (to fit)
    int const numOfQs = 2;

    // JCK samples with ordered structure (required for covariance matrix)
    Eigen::MatrixXd JCKSamplesForFit(numOfQs * N, jckNum);
    for (int i = 0; i < N; i++)
    {
        for (int q = 0; q < numOfQs; q++)
        {
            if (q == 0)
                JCKSamplesForFit.row(numOfQs * i + q) = imZBJCKs.row(i);
            else if (q == 1)
                JCKSamplesForFit.row(numOfQs * i + q) = imZSJCKs.row(i);
        }
    }

    // inverse covariance matrix blocks
    std::vector<Eigen::MatrixXd> CInvContainer(N, Eigen::MatrixXd(numOfQs, numOfQs));
    for (int i = 0; i < N; i++)
    {
        CInvContainer[i] = BlockCInverse(JCKSamplesForFit, numOfQs, i, jckNum);
    }

    // what basis functions shall be included in the fit {B, S}
    std::vector<std::pair<int, int>> BSNumbers{{1, 0}, {0, 1}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {0, 2}, {0, 3}, {3, 0}};

    // LHS matrix for the linear equation system
    Eigen::MatrixXd LHS = MatLHS(BSNumbers, muB, muS, CInvContainer, numOfQs);

    // RHS vector for the linear equation system
    Eigen::VectorXd RHS = VecRHS(BSNumbers, imZBVals, imZSVals, muB, muS, CInvContainer, numOfQs);

    // solving the linear equqation system for fitted coefficients
    Eigen::VectorXd coeffVector = (LHS).fullPivLu().solve(RHS);

/*
    // write result coefficients to screen
    std::cout << coeffVector << std::endl;

    // chi squared value
    double chiSq = ChiSq(BSNumbers, imZBVals, imZSVals, muB, muS, CInvContainer, numOfQs, coeffVector);
    // number of degrees of freedom
    int ndof = NDoF(muB, coeffVector);

    // fit quality tests
    std::cout << "\nchiSq = " << chiSq << std::endl;
    std::cout << "ndof = " << ndof << std::endl;
    std::cout << "AIC = " << AIC_weight(chiSq, ndof) << std::endl;
    std::cout << "Q = " << Q_weight(chiSq, ndof) << std::endl;
*/
    // hadrons from PDG
    std::vector<Hadron> hadronList = HadronList(PDG);
    // temperature (GeV)
    double T = .15;
    // kCut
    int kCut = 2;
    // values at mu = 0
    std::cout << "pressure: " << PressureHRG(hadronList, T, kCut) << std::endl;
    std::cout << "energy density: " << EnergyDensityHRG(hadronList, T, kCut) << std::endl;
    std::cout << "ZBB: " << SusceptibilityHRG(hadronList, 2, 0, 0, T, kCut) << std::endl;
}