source ../../setVars.sh

# ***** Install PyTorch prerequisites
installPythonModules()
{
    pip3 install torch torchvision
}

# ***** MAIN EXECUTION
installPythonModules