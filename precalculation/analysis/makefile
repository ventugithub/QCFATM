analysis.pdf: analysis.ipynb analysis.tplx
	jupyter nbconvert --to latex --template analysis.tplx analysis.ipynb
	pdflatex analysis.tex
	pdflatex analysis.tex

clean:
	rm -rf *.out *.aux *.log *.bak *.backup *.toc *~ *.ps *.dvi *.bbl *.blg *.idx *.ilg *.ind *.tex analysis_files
