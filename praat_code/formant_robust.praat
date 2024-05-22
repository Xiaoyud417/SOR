# Simple Formant Script!
#
#Xiao Yudong, 3/13/21
# Written as a demo for a lecture on Praat Scripting for LING 7030: Phonetic Theory.  Designed to provide a basis for creating other scripts.
# 
# This part presents a form to the user
form Measure Formants,Bandwidth and Duration
	comment Sound file extension:
        optionmenu file_type: 2
        option .aiff
        option .wav
endform

directory$ = chooseDirectory$ ("Choose the directory containing sound files and textgrids")
# This will need to be changed to \ below for PC users
directory$ = "'directory$'" + "/"
outdir$="'directory$'"+"formant"+"/"

# List of all the sound files in the specified directory:
Create Strings as file list... list 'directory$'*'file_type$'
number_files = Get number of strings

# This opens all the files one by one
for j from 1 to number_files
    select Strings list
    filename$ = Get string... 'j'
    Read from file... 'directory$''filename$'
    soundname$ = selected$ ("Sound")
	Create Strings as tokens... 'soundname$' "0"
	gender$ = Get string... 3
	if gender$ == "1"
		select Sound 'soundname$'
		noprogress To Formant (robust)... 0.01 5 5000 0.025 50 1.5 50 0.000001
	else
		select Sound 'soundname$'
		noprogress To Formant (robust)... 0.01 5 5500 0.025 50 1.5 50 0.000001
	endif
	select Formant 'soundname$'
	Down to Table... yes yes 6 yes 3 yes 3 yes
	select Table 'soundname$'
	Save as comma-separated file... 'outdir$''soundname$'.Table
	select all
        minus Strings list
        Remove
endfor

echo ALL FILES MEASURED
