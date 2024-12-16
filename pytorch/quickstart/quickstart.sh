source ../../setVars.sh

# ***** Run PyTorch quickstart (https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
main()
{
    echo ${cyn}#######################################################################################################${end}
    echo ${cyn}##### Run PyTorch quickstart https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#####${end}
    echo ${cyn}#######################################################################################################${end}
	echo 
	selectOption
}

trainAndSaveModel()
{
    echo ${grn}##### Train and save the model ${end}
	python3 quickstartTrainAndSave.py
}

loadAndRunModel()
{
    echo ${grn}##### Load and run the model ${end}
	python3 quickstartLoadAndRun.py
}

selectOption()
{
	echo ${grn}Select Angular environment run option : ${end}
	echo ${grn}1. Train and save model${end}
    echo ${grn}2. Load model and run prediction${end}
	read OPTION
    case $OPTION in
		1)  trainAndSaveModel
			;;
        2)  loadAndRunModel
			;;
		*) 	echo ${red}No valid option selected${end}
			selectOption
			;;
	esac
}

###### Main section
main