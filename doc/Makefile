SHELL := /bin/bash

process_version := $(addprefix tmp/, $(addsuffix .md, readme_head))
images_png := $(addprefix images/, $(addsuffix .png, structured_spherical_coords vdb_structure vklExamples))

PANDOC := pandoc
PDMINVERSION := 2000000
PDOK := $(shell expr `$(PANDOC) --version|head -n 1|cut -d' ' -f 2| sed -e 's/\.\([0-9][0-9]\)/\1/g' -e 's/\.\([0-9]\)/0\1/g' -e 's/^[0-9]\{3,4\}$$/&00/' -e 's/^[0-9]\{5,6\}$$/&00/'` \>= $(PDMINVERSION))
ifneq "$(PDOK)" "1"
  $(error You need at least pandoc v2.0)
endif

all: doc
doc: ../README.md

.PHONY: all doc clean

tmp/version: ../CMakeLists.txt
	sed -n -e "s/^project(openpgl VERSION \(.*\) LANGUAGES.*)/\1/p" $< | tr -d '\n' > $@

replace_version = sed -e "s/<OPENPGL_VERSION>/`cat tmp/version`/g" $< > $@

tmp/%.md: %.md tmp/version
	$(replace_version)

tmp/links.md: links.md

tmp/images_web.md: images.md
	sed -e "s@: @: https://openpgl.github.io/@" $< > $@

changelog.md:
	ln -s ../CHANGELOG.md $@

## ----------------------------------------------------------------------------
## Directories
## ----------------------------------------------------------------------------

$(process_version) tmp/version: | tmpdir
tmpdir:
	@mkdir -p tmp

## ----------------------------------------------------------------------------
## Markdown
## ----------------------------------------------------------------------------

../README.md: tmp/readme_head.md overview.md examples.md disclaimer.md changelog.md support.md bib.md compilation.md tmp/including.md api.md related_projects.md
	$(PANDOC) $+ -t gfm > $@

## ----------------------------------------------------------------------------
## Clean
## ----------------------------------------------------------------------------

clean:
	rm -rf tmp changelog.md __pycache__
