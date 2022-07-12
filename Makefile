.PHONY : all latex bibtex view nonstop clean distclean

TARGET=presentation
SOURCE=$(TARGET).tex
NONSTOP=-interaction nonstopmode -halt-on-error -file-line-error

all:
	pdflatex --shell-escape $(SOURCE) 
	biber 	 $(TARGET)	
	pdflatex --shell-escape $(SOURCE)
	pdflatex --shell-escape $(SOURCE)

latex:
	pdflatex --shell-escape $(SOURCE)
	pdflatex --shell-escape $(SOURCE)

biber:
	biber 	 $(TARGET)

view:
	open ./$(TARGET).pdf &

clean:
	rm *.log *.nav *.out *.snm *.toc *.fls *.fdb_latexmk *.aux *.synctex.gz *.bbl *.bcf *.blg *.run.xml
