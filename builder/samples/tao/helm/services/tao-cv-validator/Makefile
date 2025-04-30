# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

all: build_ms

SRCS:= LICENSE.txt README.md changelog.txt manifest.yaml manual_compliance_test_results.yaml \
	  $(wildcard scripts/*) $(wildcard configs/*)

build_ms: output/helm/Chart.yaml

output/helm/Chart.yaml: $(SRCS) Makefile
	ucf_ms_builder_cli service build -d . -f

clean:
	rm -rf output

