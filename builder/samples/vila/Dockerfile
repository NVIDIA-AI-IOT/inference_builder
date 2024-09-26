################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################


FROM gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:7.1.0-triton-blos-dev125 AS ds_stage

RUN apt update && apt install -y --no-install-recommends cmake git-lfs

RUN cd /opt/nvidia/deepstream/deepstream/sources \
    && git clone -b v1.1.11 --depth 1 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git \
    && cd deepstream_python_apps \
    && git submodule update --init \
    && mkdir -p bindings/build \
    && cd bindings/build \
    && cmake .. && make -j8 \
    && pip install ./pyds-1.1.11-py3-none*.whl

# Final docker base
# 24.06 does not work with DS7.0, back to 24.05
FROM nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 AS base_stage

ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

RUN apt update && apt install -y --no-install-recommends python3-gi \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-base-1.0 gir1.2-gstreamer-1.0 libyaml-cpp0.7 \
    libgstrtspserver-1.0-0 libjsoncpp25 libgles2 && \
    rm -f /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstaudioparsers.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstfaad.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvoaacenc.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstx264.so \
    /usr/lib/x86_64-linux-gnu/libavresample* /usr/lib/x86_64-linux-gnu/libavutil* \
    /usr/lib/x86_64-linux-gnu/libavcodec* /usr/lib/x86_64-linux-gnu/libavformat* \
    /usr/lib/x86_64-linux-gnu/libavfilter* /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstde265.so* \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstx265.so* /usr/lib/x86_64-linux-gnu/libde265.so* \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvpx.so* \
    /usr/lib/x86_64-linux-gnu/libmpeg2.so.0* /usr/lib/x86_64-linux-gnu/libmpeg2encpp-2.1.so* /usr/lib/x86_64-linux-gnu/libmpg123.so* \
    /usr/lib/x86_64-linux-gnu/libx265.so* /usr/lib/x86_64-linux-gnu/libx264.so* /usr/lib/x86_64-linux-gnu/libvpx.so*   \
    /usr/lib/x86_64-linux-gnu/libmpeg2convert.so* /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstopenh264.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so /usr/lib/x86_64-linux-gnu/gio/modules/libgiolibproxy.so

RUN apt remove libslurm37 libpmi2-0 -y

COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/lib /opt/nvidia/deepstream/deepstream-7.1/lib
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/bin /opt/nvidia/deepstream/deepstream-7.1/bin
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/*.sh /opt/nvidia/deepstream/deepstream-7.1/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/*.txt /opt/nvidia/deepstream/deepstream-7.1/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/README* /opt/nvidia/deepstream/deepstream-7.1/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/*.pdf /opt/nvidia/deepstream/deepstream-7.1/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/service-maker /opt/nvidia/deepstream/deepstream-7.1/service-maker

RUN /opt/nvidia/deepstream/deepstream-7.1/install.sh
RUN ln -sf deepstream-7.1 /opt/nvidia/deepstream/deepstream || true
RUN rm -rf /opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins/libnvdsgst_udp.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstchromaprint.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstmpeg2enc.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstmpeg2dec.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstmpg123.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstopenmpt.so

RUN --mount=type=bind,from=ds_stage,source=/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/build,target=/tmp/ds-bindings-build pip install /tmp/ds-bindings-build/pyds*whl --no-deps

ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/opt/tritonserver/lib/

# install trt-llm and backend
ENV SHINIT_FILE=${BASH_ENV}
RUN apt-get update && apt-get install -y --no-install-recommends rapidjson-dev python-is-python3 ccache git-lfs

# RUN pip install -U --extra-index-url https://pypi.nvidia.com --no-cache-dir --pre --no-build-isolation \
#     tensorrt-llm==0.10.0.dev2024043000 \
#     torch==2.2.2 torchvision==0.17.2 cupy-cuda12x==12.1.0

FROM base_stage AS dev_base

# VILA dependencies
RUN pip uninstall transformer-engine opencv -y
RUN pip install git+https://github.com/bfshi/scaling_on_scales.git \
    flash-attn \
    transformers==4.36.2 \
    cupy-cuda12x==12.3.0 \
    --no-build-isolation --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com

# fasttritonapi dependencies
RUN pip install tritonclient==2.45.0 inferencemodeltoolkit==0.2.27 ipdb \
    --no-cache-dir --pre --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/hw-dl-algo-bermuda-pypi/simple

RUN pip install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker-0.0.1-py3-none-linux_x86_64.whl
RUN pip install omegaconf==2.3.0
# remove unnecessary backends in case of trouble
RUN cd /opt/tritonserver/backends/ && rm -rf dali fil onnxruntime openvino pytorch repeat square tensorflow
#ENV LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorrtllm:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:${LD_LIBRARY_PATH}

# start NIM
WORKDIR /workspace

# adding more for vila-1.5

# ADD requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip uninstall -y py onnx # CVE

ENV PIP_NO_CACHE_DIR=off
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTHONUNBUFFERED=1

# Install VILA
ADD ./VILA/llava /workspace/VILA/llava
ADD ./VILA/demo_trt_llm /workspace/VILA/demo_trt_llm

RUN site_pkg_path=$(python3 -c 'import site; print(site.getsitepackages()[0])') && echo "site_pkg_path: $site_pkg_path" && cp -rv /workspace/VILA/llava/train/transformers_replace/* $site_pkg_path/transformers/
#RUN cd /workspace/VILA/demo_trt_llm && sh apply_patch.sh
ENV PYTHONPATH="/workspace/VILA:${PYTHONPATH}"

# triton model repo
ENV MODEL_NAME="vila"

# orignal model dir
ENV VILA_MODEL_NAME="vila1.5-13b"

# CVE
RUN pip install aiohttp==3.9.4 pyarrow==16.1.0 --no-cache-dir --pre
RUN apt update && apt install -y --no-install-recommends openssh-client=1:8.9p1-3ubuntu0.10

FROM dev_base AS api_server

# CVE
RUN pip uninstall -y onnx jupyterlab Werkzeug
RUN rm -rf /usr/local/nvm/versions/node \
    /opt/pytorch/pytorch/third_party/onnx/onnx

# triton:24.06 TRT version not compatible with DS7.0
# RUN rm -rf /opt/nvidia/deepstream/deepstream-7.0/lib/gst-plugins/libnvdsgst_infer.so \
#     /opt/nvidia/deepstream/deepstream-7.0/lib/gst-plugins/libnvdsgst_osd.so \
#     /opt/nvidia/deepstream/deepstream-7.0/lib/gst-plugins/libnvdsgst_deepstream_bins.so \
#     /opt/nvidia/deepstream/deepstream-7.0/lib/gst-plugins/libnvdsgst_inferaudio.so \
#     /opt/nvidia/deepstream/deepstream-7.0/lib/gst-plugins/libnvdsgst_infer.so \
#     /opt/nvidia/deepstream/deepstream-7.0/lib/gst-plugins/libnvdsgst_inferserver.so \
#     /opt/nvidia/deepstream/deepstream-7.0/lib/gst-plugins/libnvdsgst_osd.so

ENV CHECKPOINTS_DIR=/workspace/checkpoints
ENV NVCF_METER_EVENT=AIPlayground_Inference

ADD ./common /workspace/common
ADD ./lib /workspace/lib
ADD ./custom /workspace/custom
ADD ./model_repo /workspace/model_repo
ADD ./server_run.sh /workspace/
ADD ./inference.py /workspace/
ADD ./data_model.py /workspace/


ENV PYTHONPATH="/workspace/VILA:/workspace:${PYTHONPATH}"

#ENTRYPOINT ["bash", "/workspace/server_run.sh"]
CMD bash /workspace/server_run.sh


FROM dev_base AS trt_optimize
ENV CHECKPOINTS_DIR=/workspace/checkpoints

ADD ./build_visual_engine.py /workspace/VILA/demo_trt_llm/build_visual_engine.py
ADD ./build_visual_engine_from_onnx.py /workspace/VILA/demo_trt_llm/build_visual_engine_from_onnx.py

# ADD ./quantize.py /workspace/
ADD ./trt_optimize.sh /workspace/

#ENTRYPOINT ["bash", "/workspace/trt_optimize.sh"]
CMD bash /workspace/trt_optimize.sh

