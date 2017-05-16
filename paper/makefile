PROJECT_NAME=atm
${PROJECT_NAME}: ${PROJECT_NAME}*.tex ${PROJECT_NAME}.bib sections/*.tex
	pdflatex ${PROJECT_NAME}.tex
	bibtex ${PROJECT_NAME}
	pdflatex ${PROJECT_NAME}.tex
	pdflatex ${PROJECT_NAME}.tex

clean:
	rm -f *.out *.aux *.log *.bak *.backup *.toc *~ *.ps *.dvi *.bbl *.blg *.idx *.ilg *.ind *.lof
	rm ${PROJECT_NAME}Notes.bib

