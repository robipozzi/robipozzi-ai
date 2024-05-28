##### Terminal Colors - START
red=$'\e[1;31m'
grn=$'\e[1;32m'
yel=$'\e[1;33m'
blu=$'\e[1;34m'
mag=$'\e[1;35m'
cyn=$'\e[1;36m'
end=$'\e[0m'
coffee=$'\xE2\x98\x95'
coffee3="${coffee} ${coffee} ${coffee}"
##### Terminal Colors - END

###### Variable section - START
NLP_TECHNIQUE_SCRIPT=$1
NLP_SCRIPT=
###### Variable section - END

# ***** Function section - START
main()
{
	if [ -z $NLP_TECHNIQUE_SCRIPT ]; then 
        printSelectNLPTechnique
    fi
	runNLPScript
}

###############
## printHelp ##
###############
printHelp()
{
	printf "\n${yel}Usage:${end}\n"
  	printf "${cyn}$SCRIPT <NLP_TECHNIQUE_SCRIPT>${end}\n"
	printf "${cyn}where:${end}\n"
	printf "${cyn}- <NLP_TECHNIQUE_SCRIPT> can be one of the following${end}\n"
	printf "${cyn}	1. Tokenization${end}\n"
	printf "${cyn}	2. Basic Preprocessing${end}\n"
	printf "${cyn}	3. Advanced Preprocessing${end}\n"
    printf "${cyn}	4. Basic Bag-of-Words${end}\n"
    printf "${cyn}	5. TF-IDF (Term Frequency & Inverse Document Frequency)${end}\n"
}

printSelectNLPTechnique()
{
	echo ${grn}Select Kafka cluster run platform : ${end}
    echo "${grn}1. Tokenization${end}"
    echo "${grn}2. Basic Preprocessing${end}"
	echo "${grn}3. Advanced Preprocessing${end}"
    echo "${grn}4. Basic Bag-of-Words${end}"
    echo "${grn}5. TF-IDF (Term Frequency & Inverse Document Frequency)${end}"
	read NLP_TECHNIQUE_SCRIPT
	setNLPScript
}

setNLPScript()
{
	case $NLP_TECHNIQUE_SCRIPT in
		1)  NLP_SCRIPT="1-tokenization.py"
			;;
        2)  NLP_SCRIPT="2-basicPreProcessing.py"
            ;;
		3)  NLP_SCRIPT="3-advancedPreProcessing.py"
            ;;
        4)  NLP_SCRIPT="4-basicBagOfWords.py"
            ;;
        5)  NLP_SCRIPT="5-tfidf.py"
            ;;
		*) 	printf "\n${red}No valid option selected${end}\n"
			printSelectNLPTechnique
			;;
	esac
}

runNLPScript()
{
    python3 $NLP_SCRIPT
}
# ***** Function section - END

# ##############################################
# #################### MAIN ####################
# ##############################################
# ************ START evaluate args ************"
if [ "$1" != "" ]; then
    setNLPScript
fi
# ************** END evaluate args **************"
RUN_FUNCTION=main
$RUN_FUNCTION