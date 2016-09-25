PROJECT_NAME=report_july_2016

${PROJECT_NAME}: ${PROJECT_NAME}*.tex
	pdflatex ${PROJECT_NAME}.tex
	bibtex ${PROJECT_NAME}
	pdflatex ${PROJECT_NAME}.tex
	pdflatex ${PROJECT_NAME}.tex

clean:
	rm -f *.out *.aux *.log *.bak *.backup *.toc *~ *.ps *.dvi *.bbl *.blg *.idx *.ilg *.ind *.fls *.fdb_latexmk *Notes.bib *.synctex.gz *.xml *-blx.bib *.soc
