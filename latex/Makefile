# Build the three PDFs from shared sources and copy them to results/papers/.
#   make all        (= tables ieee report springer, then copy)
#   make clean
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error
PAPERS  = ../results/papers

.PHONY: all tables ieee springer report copy clean
all: tables ieee report springer copy

tables:
	python3 tools/make_tables.py

ieee:
	cd ieee && $(LATEXMK) main.tex

springer:
	cd springer && $(LATEXMK) main.tex

report:
	cd report && $(LATEXMK) main.tex

copy:
	mkdir -p $(PAPERS)
	cp ieee/main.pdf $(PAPERS)/ieee_quhi.pdf
	cp springer/main.pdf $(PAPERS)/springer_quhi.pdf
	cp report/main.pdf $(PAPERS)/quhi_full_report.pdf

clean:
	for d in ieee springer report; do (cd $$d && latexmk -C main.tex); done
