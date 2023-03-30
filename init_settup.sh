# cd to folder
cd /data/s4320200
mkdir getafixtest
cd getafixtest
echo "created directory"

# run git clone
git config --global user.name HughMcDougall
git config --global user.email token
git clone https://github.com/HughMcDougall/GetafixTest
#	Will be prompted for username & token here, or at least token
echo "git clone successful"

# create conda environment for running script
conda create --name getafix_test_conda python=3.10 pip numpy matplotlib astropy scipy
echo "conda environment created"