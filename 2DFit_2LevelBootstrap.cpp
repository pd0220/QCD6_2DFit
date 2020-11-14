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
// argv[5] --> bootstrap sample number
int main(int argc, char **argv)
{
    // check argument list
    if (argc < 6)
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
    // number of bootstrap samples to use
    int const bsNum = std::atoi(argv[5]);

    // generate random numbers
    std::random_device rd{};
    std::mt19937 gen(rd());
    // random generator lambda
    auto rand = [&gen](int const &numOfBlocks) {
        std::uniform_int_distribution<> distr(0, numOfBlocks - 1);
        return (int)distr(gen);
    };

    // data at mu = 0
    Eigen::MatrixXd const rawDataMatMuZero = ReadFile("../muZero.txt");

    // new matrix for the fit data
    Eigen::MatrixXd FitMat = Eigen::MatrixXd::Zero(rawDataMat.rows() + 1, rawDataMat.cols());
    FitMat.bottomRows(rawDataMat.rows()) = rawDataMat;
    // beta and Nt (not relevant now)
    FitMat(0, 0) = rawDataMatMuZero(0, 0);
    FitMat(0, 1) = rawDataMatMuZero(0, 1);
    // muB = 0
    FitMat(0, 2) = 0.;
    // muS = 0
    FitMat(0, 3) = 0.;
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
    // new size for Z to contain bootsrap samples
    int const ZSize_NEW = 2 + bsNum;
    // vectors to store imZB values and their estimated errors (results) + bootstrap
    Eigen::VectorXd imZBVals(rows);
    Eigen::VectorXd imZBErrs(rows);
    Eigen::MatrixXd imZB_boot(rows, bsNum);
    // vectors to store imZQ values and their estimated errors (results) + bootstrap
    Eigen::VectorXd imZQVals(rows);
    Eigen::VectorXd imZQErrs(rows);
    Eigen::MatrixXd imZQ_boot(rows, bsNum);
    // vectors to store imZS values and their estimated errors (results) + bootstrap
    Eigen::VectorXd imZSVals(rows);
    Eigen::VectorXd imZSErrs(rows);
    Eigen::MatrixXd imZS_boot(rows, bsNum);
    // vectors to store ZBB values and their estimated errors (results) + bootstrap
    Eigen::VectorXd ZBBVals(rows);
    Eigen::VectorXd ZBBErrs(rows);
    Eigen::MatrixXd ZBB_boot(rows, bsNum);
    // vectors to store ZQQ values and their estimated errors (results) + bootstrap
    Eigen::VectorXd ZQQVals(rows);
    Eigen::VectorXd ZQQErrs(rows);
    Eigen::MatrixXd ZQQ_boot(rows, bsNum);
    // vectors to store ZSS values and their estimated errors (results) + bootstrap
    Eigen::VectorXd ZSSVals(rows);
    Eigen::VectorXd ZSSErrs(rows);
    Eigen::MatrixXd ZSS_boot(rows, bsNum);
    // vectors to store ZBQ values and their estimated errors (results) + bootstrap
    Eigen::VectorXd ZBQVals(rows);
    Eigen::VectorXd ZBQErrs(rows);
    Eigen::MatrixXd ZBQ_boot(rows, bsNum);
    // vectors to store ZBS values and their estimated errors (results) + bootsrap
    Eigen::VectorXd ZBSVals(rows);
    Eigen::VectorXd ZBSErrs(rows);
    Eigen::MatrixXd ZBS_boot(rows, bsNum);
    // vectors to store ZQS values and their estimated errors (results) + bootsrap
    Eigen::VectorXd ZQSVals(rows);
    Eigen::VectorXd ZQSErrs(rows);
    Eigen::MatrixXd ZQS_boot(rows, bsNum);
    // vectors to store ZII values and their estimated errors (results) + bootsrap
    Eigen::VectorXd ZIIVals(rows);
    Eigen::VectorXd ZIIErrs(rows);
    Eigen::MatrixXd ZII_boot(rows, bsNum);
    // container for "flavour vectors"
    std::vector<Eigen::VectorXd> ZContainer(ZNum);
    // loop for every row
    for (int i = 0; i < rows; i++)
    {
        // filling up vectors
        for (int j = 0; j < ZNum; j++)
        {
            // create temporary vector for sample number reduction
            Eigen::VectorXd tmpVec = FitMat.row(i).segment(4 + j * ZSize, ZSize);

            // temporary JCK vectors
            Eigen::VectorXd tmpJCKVec_OLD = tmpVec.segment(2, ZSize - 2);
            // calculate original blocks and perform the sample number reduction
            Eigen::VectorXd tmpNewBlocks = ReducedBlocks(tmpJCKVec_OLD, divisor);

            // JCK result (+ val + err)
            // results
            Eigen::VectorXd tmpResult(ZSize_NEW);
            // value
            tmpResult(0) = tmpVec(0);
            // error
            tmpResult(1) = tmpVec(1);
            // bootstrap
            Eigen::VectorXd tmpBootstrapSamples(bsNum);
            for (int iBS = 0; iBS < bsNum; iBS++)
            {
                tmpBootstrapSamples(iBS) = BootstrapSamplesCalculation(tmpNewBlocks, rand);
            }
            tmpResult.segment(2, bsNum) = tmpBootstrapSamples;

            ZContainer[j] = tmpResult;
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
        imZB_boot.row(i) = imZB.segment(2, bsNum);
        // imZQ
        imZQ_boot.row(i) = imZQ.segment(2, bsNum);
        // imZS
        imZS_boot.row(i) = imZS.segment(2, bsNum);
        // ZBB
        ZBB_boot.row(i) = ZBB.segment(2, bsNum);
        // ZQQ
        ZQQ_boot.row(i) = ZQQ.segment(2, bsNum);
        // ZSS
        ZSS_boot.row(i) = ZSS.segment(2, bsNum);
        // ZBQ
        ZBQ_boot.row(i) = ZBQ.segment(2, bsNum);
        // ZBS
        ZBS_boot.row(i) = ZBS.segment(2, bsNum);
        // ZQS
        ZQS_boot.row(i) = ZQS.segment(2, bsNum);
        // ZII
        ZII_boot.row(i) = ZII.segment(2, bsNum);

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

    // tuple container for fitted coefficient sets with maps of BS pairs and bootstrap samples (= bootstrap coefficients)
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
    std::vector<Eigen::MatrixXd> bootstrapSampleContainer{imZB_boot, imZS_boot, ZBB_boot, ZBS_boot, ZSS_boot};
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
        // bootstrap samples (required for covariance matrix estimation)
        Eigen::MatrixXd bootstrapSamplesForFit = bootstrapSampleContainer[iFit];
        // sector to include in fit
        std::vector<std::pair<int, int>> BSNumbers = sectorContainer[iFit];
        // derivatives to include in basis function
        std::vector<std::pair<int, int>> DOrders = dOrdersContainer[iFit];

        // number of sectors
        int sectorNumber = static_cast<int>(BSNumbers.size());

        // inverse covariance matrix blocks
        std::vector<Eigen::MatrixXd> CInvContainer(N, Eigen::MatrixXd(numOfQs, numOfQs));

        for (int i = 0; i < N; i++)
        {
            CInvContainer[i] = BlockCInverseBootstrap(bootstrapSamplesForFit, numOfQs, i, bsNum);
        }

        // LHS matrix for the linear equation system
        Eigen::MatrixXd LHS = MatLHS(BSNumbers, DOrders, muB, muS, CInvContainer);

        // RHS vector for the linear equation system
        Eigen::VectorXd RHS = VecRHS(BSNumbers, DOrders, yMat, muB, muS, CInvContainer);

        // solving the linear equqation system for fitted coefficients
        Eigen::VectorXd coeffVector = (LHS).fullPivLu().solve(RHS);

        // fit for bootstrap samples
        std::vector<Eigen::VectorXd> bootstrap_RHS(bsNum);
        for (int i = 0; i < bsNum; i++)
        {
            // y data matrix for bootstrap fits
            Eigen::MatrixXd yMatBootstrap(numOfQs, N);
            yMatBootstrap.row(0) = bootstrapSamplesForFit.col(i);
            // RHS vectors from bootsrap samples
            bootstrap_RHS[i] = VecRHS(BSNumbers, DOrders, yMatBootstrap, muB, muS, CInvContainer);
        }

        // fit with bootstrap samples
        Eigen::MatrixXd bootstrap_coeffVector(coeffVector.size(), bsNum);
        for (int i = 0; i < bsNum; i++)
        {
            bootstrap_coeffVector.col(i) = (LHS).fullPivLu().solve(bootstrap_RHS[i]);
        }

        // making element for coefficient container
        std::tuple<Eigen::VectorXd, std::vector<std::pair<int, int>>, Eigen::MatrixXd> tupleContainer{coeffVector, BSNumbers, bootstrap_coeffVector};
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
        // bootstrap samples for {B, S} sectors
        std::vector<Eigen::VectorXd> BS_boot{};

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
                    // add bootstrap samples to container
                    BS_boot.push_back(std::get<2>(coeffContainer[iData]).row(iSector));
                }
            }
        }

        // y data to calculate RHS vector
        Eigen::VectorXd yVec(BSOccurences);
        // BS samples (required for covariance matrix estimation)
        Eigen::MatrixXd bootstrapSamplesForFit(BSOccurences, bsNum);
        for (int i = 0; i < BSOccurences; i++)
        {
            yVec(i) = BSCoeffs[i];
            bootstrapSamplesForFit.row(i) = BS_boot[i];
        }

        // inverse covariance matrix
        Eigen::MatrixXd CInv = BlockCInverseBootstrap(bootstrapSamplesForFit, BSOccurences, 0, bsNum);

        // basis function is constant
        Eigen::VectorXd basisConstant = Eigen::VectorXd::Constant(BSOccurences, 1);

        // LHS matrix for the linear equation system
        Eigen::MatrixXd LHS = basisConstant.transpose() * CInv * basisConstant;

        // RHS vector for the linear equation system
        Eigen::MatrixXd RHS = yVec.transpose() * CInv * basisConstant;

        // error estimation via jackknife method
        std::vector<Eigen::VectorXd> bootstrap_RHS(bsNum);
        for (int i = 0; i < bsNum; i++)
        {
            // RHS vectors from jackknife samples
            bootstrap_RHS[i] = bootstrapSamplesForFit.col(i).transpose() * CInv * basisConstant;
        }

        // fit with bootstrap samples
        std::vector<Eigen::VectorXd> bootstrap_coeffVector(bsNum);
        for (int i = 0; i < bsNum; i++)
        {
            bootstrap_coeffVector[i] = (LHS).fullPivLu().solve(bootstrap_RHS[i]);
        }
        // estimate error from jackknife fits (divided by square root of block numbers ~ number of jaccknife samples divided by divisor for sample number reduction)
        Eigen::VectorXd errorVec = JCKFitErrorEstimation((LHS).fullPivLu().solve(RHS), bootstrap_coeffVector) / std::sqrt(jckNum);

        // fitted coefficient
        std::cout << "{" << FullSectors[iFit].first << ", " << FullSectors[iFit].second << "} " << (LHS).fullPivLu().solve(RHS) << " +/- " << errorVec << std::endl;
    }
}
