// including used header
#include "AnalysisTools.hh"

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
    // number of cols and rows of raw data matrix
    //int const cols = rawDataMat.cols();
    int const rows = rawDataMat.rows();

    // chemical potentials for baryon numbers and strangeness
    Eigen::VectorXd const muB = rawDataMat.col(2);
    Eigen::VectorXd const muS = rawDataMat.col(3);

    // susceptibilities (regarding u, d, s, c flavours) with error and jackknife samples
    // size of vectors (val + err + jck samples)
    int const ZSize = 2 + jckNum;
    // overwrite jackknife sample number for sample number reduction
    jckNum = jckNum / divisor;

    // container for "flavour vectors"
    std::vector<Eigen::VectorXd> ZContainer(ZNum);
    // imZu is zero at mu = 0
    ZContainer[0] = Eigen::VectorXd::Zero(ZSize);
    // imZs is zero at mu = 0
    ZContainer[1] = Eigen::VectorXd::Zero(ZSize);

    // loop for every row
    for (int i = 0; i < rows; i++)
    {
        // filling up vectors
        for (int j = 0; j < ZNum - 2; j++)
        {
            // if sample number reduction is ON
            if (std::abs(1 - divisor) > eps)
            {
                // create temporary vector for sample number reduction
                Eigen::VectorXd tmpVec = rawDataMat.row(i).segment(2 + j * ZSize, ZSize);

                // temporary JCK vectors
                Eigen::VectorXd tmpJCKVec_OLD = tmpVec.segment(2, ZSize - 2);
                // calculate original blocks and perform the sample number reduction
                Eigen::VectorXd tmpJCKVec_NEW = JCKSamplesCalculation(JCKReducedBlocks(tmpJCKVec_OLD, divisor));

                // JCK result (+ val + err)
                // new size for Z
                int ZSize_NEW = 2 + jckNum;
                // results
                Eigen::VectorXd tmpResult(ZSize_NEW);
                // value
                tmpResult(0) = tmpVec(0);
                // error
                tmpResult(1) = tmpVec(1);
                // jackknife samples
                tmpResult.segment(2, jckNum) = tmpJCKVec_NEW;

                ZContainer[j + 2] = tmpResult;
            }
            else
                ZContainer[j + 2] = rawDataMat.row(i).segment(2 + j * ZSize, ZSize);
        }
    }

    // test
    std::cout << ZContainer[30] << std::endl;
}