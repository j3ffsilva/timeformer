#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 path/to/paper.tex" >&2
  exit 2
fi

tex_file="$1"

if [ ! -f "$tex_file" ]; then
  echo "File not found: $tex_file" >&2
  exit 1
fi

case "$tex_file" in
  *.tex) ;;
  *)
    echo "Input file must have a .tex extension: $tex_file" >&2
    exit 2
    ;;
esac

paper="$(basename "$tex_file" .tex)"
source_dir="$(dirname "$tex_file")"
out_dir="out"

mkdir -p "$out_dir"

export TEXINPUTS="templates/LaTeX2e//:${source_dir}//:"
export BIBINPUTS="${source_dir}//:"

pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$out_dir" "$tex_file"
bibtex "$out_dir/$paper"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$out_dir" "$tex_file"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$out_dir" "$tex_file"

cp "$out_dir/$paper.pdf" "$paper.pdf"
cp "$out_dir/$paper.log" "$paper.log"

find "$out_dir" -maxdepth 1 -type f -name "$paper.*" -delete
find . -maxdepth 1 -type f \( \
  -name "$paper.aux" -o \
  -name "$paper.toc" -o \
  -name "$paper.out" -o \
  -name "$paper.bbl" -o \
  -name "$paper.blg" -o \
  -name "$paper.fls" -o \
  -name "$paper.fdb_latexmk" -o \
  -name "$paper.synctex.gz" \
\) -delete

echo "Wrote $paper.pdf and $paper.log"
