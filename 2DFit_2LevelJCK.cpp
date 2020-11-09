// including used header
#include "AnalysisTools.hh"

// ------------------------------------------------------------------------------------------------------------

// PDG file name (hadron list)
std::string const PDG = "../PDG.txt";

// ------------------------------------------------------------------------------------------------------------

// main function
// argv[1] --> name of given file with dataset
// argv[2] --> number of jackknife samples
// argv[3] --> number of used susceptibilities (Zu, Zs, etc.)
// argv[4] --> divisor for jackknife sample number reduction
int main(int argc, char **argv)
{
    // check argument list
    if (argc < 5)
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
    int jckNum = std::atoi(argv[2]);
    // number of used susceptibilities
    int const ZNum = std::atoi(argv[3]);
    // divisor for jackknife sample number reduction
    int const divisor = std::atoi(argv[4]);
    // check if the number of jackknife samples can be divided by the divisor
    if (jckNum % divisor > eps)
    {
        std::cout << "ERROR\nThe ,,jckNum'' and ,,divisor'' pair is not appropriate." << std::endl;
        std::exit(-1);
    }

    // data at mu = 0
    Eigen::MatrixXd const rawDataMatMuZero = ReadFile("../muZero.txt");

    // new matrix for the fit data
    Eigen::MatrixXd FitMat = Eigen::MatrixXd::Zero(rawDataMat.rows() + 1, rawDataMat.cols());
    FitMat.bottomRows(rawDataMat.rows()) = rawDataMat;
    // beta and Nt (not relevant now)
    FitMat(0, 0) = rawDataMatMuZero(0, 0);
    FitMat(0, 1) = rawDataMatMuZero(0, 1);
    // muB = 0
    FitMat(0, 2) = 0;
    // muS = 0
    FitMat(0, 3) = 0;
    // non-zero data (imZu = 0 and imZs = 0 at mu = 0)
    FitMat.row(0).segment(4 + 2 * (jckNum + 2), (ZNum - 2) * (jckNum + 2)) = rawDataMatMuZero.row(0).segment(2, (ZNum - 2) * (jckNum + 2));

    // number of rows of fit matrix
    int const rows = FitMat.rows();

    // chemical potentials for baryon numbers and strangeness
    Eigen::VectorXd const muB = FitMat.col(2);
    Eigen::VectorXd const muS = FitMat.col(3);

    // susceptibilities (regarding u, d, s flavours) with error and jackknife samples
    // size of vectors (val + err + jck samples)
    int const ZSize = 2 + jckNum;
    // overwrite jackknife sample number for sample number reduction
    jckNum = jckNum / divisor;
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
            // if sample number reduction is ON
            if (std::abs(1 - divisor) > eps)
            {
                // create temporary vector for sample number reduction
                Eigen::VectorXd tmpVec = FitMat.row(i).segment(4 + j * ZSize, ZSize);

                // temporary JCK vectors
                Eigen::VectorXd tmpJCKVec_OLD = tmpVec.segment(2, ZSize - 2);
                // calculate original blocks and perform the sample number reduction
                Eigen::VectorXd tmpJCKVec_NEW = JCKSamplesCalculation(ReducedBlocks(tmpJCKVec_OLD, divisor));

                // JCK result (+ val + err)
                // new size for Z
                int ZSize_NEW = 2 + jckNum;
                // results
                Eigen::VectorXd tmpResult(ZSize_NEW);
                // value
                tmpResult(0) = tmpVec(0);
                // error
                tmpResult(1) = tmpVec(1);
                // jackknife
                tmpResult.segment(2, jckNum) = tmpJCKVec_NEW;

                ZContainer[j] = tmpResult;
            }
            else
                ZContainer[j] = FitMat.row(i).segment(4 + j * ZSize, ZSize);
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

        // save errors
        // imZB
        imZBErrs(i) = ZError(imZB);
        // imZQ
        imZQErrs(i) = ZError(imZQ);
        // imZS
        imZSErrs(i) = ZError(imZS);
        // ZBB
        ZBBErrs(i) = ZError(ZBB);
        // ZQQ
        ZQQErrs(i) = ZError(ZQQ);
        // ZSS
        ZSSErrs(i) = ZError(ZSS);
        // ZBQ
        ZBQErrs(i) = ZError(ZBQ);
        // ZBS
        ZBSErrs(i) = ZError(ZBS);
        // ZQS
        ZQSErrs(i) = ZError(ZQS);
        // ZII
        ZIIErrs(i) = ZError(ZII);
    }

    //
    // START FIT
    // --> imZB, imZS, ZBB, ZBS and ZSS (uncorrelated)
    // making fits seperately first
    //

    // number of x-values (muB and muS)
    int const N = muB.size();

    // number of quantitites
    int const numOfQs = 1;

    // tuple container for fitted coefficient sets with maps of BS pairs and jackknife samples
    std::vector<std::tuple<Eigen::VectorXd, std::vector<std::pair<int, int>>, Eigen::MatrixXd>> coeffContainer{};

    // sectors: what basis functions shall be included in the fit {B, S}
    // full is {1, 0}, {0, 1}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {0, 2}, {0, 3}, {3, 0}
    std::vector<std::pair<int, int>> FullSectors{{1, 0}, {0, 1}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {0, 2}, {0, 3}, {3, 0}};
    std::vector<std::pair<int, int>> imZBSectors{{1, 0}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {3, 0}};
    std::vector<std::pair<int, int>> imZSSectors{{0, 1}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {2, 3}, {0, 2}, {0, 3}};
    std::vector<std::pair<int, int>> ZBBSectors{{1, 0}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {3, 0}};
    std::vector<std::pair<int, int>> ZBSSectors{{1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {2, 3}};
    std::vector<std::pair<int, int>> ZSSSectors{{0, 1}, {1, -1}, {1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {2, 3}, {0, 2}, {0, 3}};

    // what order of derivatives are int the basis functions {B, S}
    std::vector<std::pair<int, int>> imZBDOrders{{1, 0}};
    std::vector<std::pair<int, int>> imZSDOrders{{0, 1}};
    std::vector<std::pair<int, int>> ZBBDOrders{{2, 0}};
    std::vector<std::pair<int, int>> ZBSDOrders{{1, 1}};
    std::vector<std::pair<int, int>> ZSSDOrders{{0, 2}};

    // data container
    std::vector<Eigen::VectorXd> dataContainer{imZBVals, imZSVals, ZBBVals, ZBSVals, ZSSVals};
    // jackknife sample container
    std::vector<Eigen::MatrixXd> jckSampleContainer{imZBJCKs, imZSJCKs, ZBBJCKs, ZBSJCKs, ZSSJCKs};
    // sector container
    std::vector<std::vector<std::pair<int, int>>> sectorContainer{imZBSectors, imZSSectors, ZBBSectors, ZBSSectors, ZSSSectors};
    // derivative order container
    std::vector<std::vector<std::pair<int, int>>> dOrdersContainer{imZBDOrders, imZSDOrders, ZBBDOrders, ZBSDOrders, ZSSDOrders};

    // calculating uncorrelated fits for susceptibilities
    for (int iFit = 0; iFit < 5; iFit++)
    {
        // y data matrix to calculate RHS vector
        Eigen::MatrixXd yMat(numOfQs, N);
        yMat.row(0) = dataContainer[iFit];
        // JCK samples (required for covariance matrix estimation)
        Eigen::MatrixXd JCKSamplesForFit = jckSampleContainer[iFit];
        // sector to include in fit
        std::vector<std::pair<int, int>> BSNumbers = sectorContainer[iFit];
        // derivatives to include in basis function
        std::vector<std::pair<int, int>> DOrders = dOrdersContainer[iFit];

        // number of sectors
        int sectorNumber = static_cast<int>(BSNumbers.size());

        // inverse covariance matrix blocks
        std::vector<Eigen::MatrixXd> CInvContainer(N, Eigen::MatrixXd::Zero(numOfQs, numOfQs));

        for (int i = 0; i < N; i++)
        {
            CInvContainer[i] = BlockCInverseJCK(JCKSamplesForFit, numOfQs, i, jckNum);
        }

        // LHS matrix for the linear equation system
        Eigen::MatrixXd LHS = MatLHS(BSNumbers, DOrders, muB, muS, CInvContainer);

        // RHS vector for the linear equation system
        Eigen::VectorXd RHS = VecRHS(BSNumbers, DOrders, yMat, muB, muS, CInvContainer);

        // solving the linear equqation system for fitted coefficients
        Eigen::VectorXd coeffVector = (LHS).fullPivLu().solve(RHS);

        // fit for jackknife samples
        std::vector<Eigen::VectorXd> JCK_RHS(jckNum);
        for (int i = 0; i < jckNum; i++)
        {
            // y data matrix for jackknife fits
            Eigen::MatrixXd yMatJCK(numOfQs, N);
            yMatJCK.row(0) = JCKSamplesForFit.col(i);
            // RHS vectors from jackknife samples
            JCK_RHS[i] = VecRHS(BSNumbers, DOrders, yMatJCK, muB, muS, CInvContainer);
        }

        // fit with jackknife samples
        Eigen::MatrixXd JCK_coeffVector(coeffVector.size(), jckNum);
        for (int i = 0; i < jckNum; i++)
        {
            JCK_coeffVector.col(i) = (LHS).fullPivLu().solve(JCK_RHS[i]);
        }

        // making element for coefficient container
        std::tuple<Eigen::VectorXd, std::vector<std::pair<int, int>>, Eigen::MatrixXd> tupleContainer{coeffVector, BSNumbers, JCK_coeffVector};
        // add to container
        coeffContainer.push_back(tupleContainer);
    }

    // fits for all the fitted coefficients
    for (int iFit = 0; iFit < FullSectors.size(); iFit++)
    {
        // count number of given {B, S} sector occurences
        int BSOccurences = 0;
        // fitted parameters for {B, S} sectors
        std::vector<double> BSCoeffs{};
        // jackknife samples for {B, S} sectors
        std::vector<Eigen::VectorXd> BSJCKs{};

        // loop through imZB, imZS, ZBB, ZBS, ZSS fit conatiners
        for (int iData = 0; iData < static_cast<int>(coeffContainer.size()); iData++)
        {
            // loop through fitted sector coeffiticents
            for (int iSector = 0; iSector < static_cast<int>(std::get<0>(coeffContainer[iData]).size()); iSector++)
            {
                // check if given sector is present in fit
                if (FullSectors[iFit] == std::get<1>(coeffContainer[iData])[iSector])
                {
                    // increase value of occurence number
                    BSOccurences++;
                    // add fitted coefficient to container
                    BSCoeffs.push_back(std::get<0>(coeffContainer[iData])[iSector]);
                    // add jackknife samples to container
                    BSJCKs.push_back(std::get<2>(coeffContainer[iData]).row(iSector));
                }
            }
        }

        // y data to calculate RHS vector
        Eigen::VectorXd yVec(BSOccurences);
        // JCK samples (required for covariance matrix estimation)
        Eigen::MatrixXd JCKSamplesForFit(BSOccurences, jckNum);
        for (int i = 0; i < BSOccurences; i++)
        {
            yVec(i) = BSCoeffs[i];
            JCKSamplesForFit.row(i) = BSJCKs[i];
        }

        // inverse covariance matrix
        Eigen::MatrixXd CInv = BlockCInverseJCK(JCKSamplesForFit, BSOccurences, 0, jckNum);

        // basis function is constant
        Eigen::VectorXd basisConstant = Eigen::VectorXd::Constant(BSOccurences, 1);

        // LHS matrix for the linear equation system
        Eigen::MatrixXd LHS = basisConstant.transpose() * CInv * basisConstant;

        // RHS vector for the linear equation system
        Eigen::MatrixXd RHS = yVec.transpose() * CInv * basisConstant;

        // error estimation via jackknife method
        std::vector<Eigen::VectorXd> JCK_RHS(jckNum);
        for (int i = 0; i < jckNum; i++)
        {
            // RHS vectors from jackknife samples
            JCK_RHS[i] = JCKSamplesForFit.col(i).transpose() * CInv * basisConstant;
        }

        // fit with jackknife samples
        std::vector<Eigen::VectorXd> JCK_coeffVector(jckNum);
        for (int i = 0; i < jckNum; i++)
        {
            JCK_coeffVector[i] = (LHS).fullPivLu().solve(JCK_RHS[i]);
        }
        // estimate error from jackknife fits
        Eigen::VectorXd errorVec = JCKFitErrorEstimation((LHS).fullPivLu().solve(RHS), JCK_coeffVector);

        // fitted coefficient
        std::cout << "{" << FullSectors[iFit].first << ", " << FullSectors[iFit].second << "} " << (LHS).fullPivLu().solve(RHS) << " +/- " << errorVec << std::endl;
    }
}
