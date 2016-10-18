PDFs=report_october_2016.pdf

default:$(PDFs)

%.pdf: %.tex
	latexmk -pdf $^

clean:
	rm -f *.out *.aux *.log *.bak *.backup *.toc *~ *.ps *.dvi *.bbl *.blg *.idx *.ilg *.ind *.fls *.fdb_latexmk *Notes.bib *.synctex.gz *.xml *-blx.bib *.soc
