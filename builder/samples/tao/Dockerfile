# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


FROM "nvcr.io/nvidia/tritonserver-pb24h1:24.03.04-py3" AS nim_builder_base

ARG DS_PKG=deepstream_sdk_v7.1.0_x86_64.tbz2

ENV NIM_DIR_PATH="/opt/nim" \
    PIP_INDEX_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3-distutils \
    && if ! command -v pip &> /dev/null; then \
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
        && python3 get-pip.py \
        && rm get-pip.py; \
    else \
        python3 -m pip install --upgrade pip; \
    fi

    #TODO: remove curl?
    #&& apt-get purge -y --auto-remove curl

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nimlib[triton,ngcsdk]==0.1.47

LABEL com.nvidia.nim.base_image="nvcr.io/nvidia/tritonserver-pb24h1:24.03.04-py3"
LABEL com.nvidia.nim.name={{MODEL_NAME}}
LABEL com.nvidia.nim.type=triton
LABEL com.nvidia.nim.version=0.0.1
LABEL com.nvidia.nim.nspect=NSPECT-Z39R-IVVG
LABEL com.nvidia.nim.inference_protocol=http

ENV NIM_CACHE_PATH="/opt/nim/.cache" \
    NIM_NAME=nv-tao-inference \
    PYTHONUNBUFFERED=1 \
    NGC_API_KEY=

ENV BACKEND_TYPE=triton
ENV BASE_IMAGE="nvcr.io/nvidia/tritonserver-pb24h1:24.03.04-py3"
ENV NIMTOOLS_VERSION=0.3.1rc7
ENV BACKEND_TYPE="triton"
ENV NIM_NSPECT_ID=NSPECT-Z39R-IVVG
ENV NIM_MODEL_NAME={{MODEL_NAME}}
ENV NIM_LOG_LEVEL="DEBUG"

# COPY ./model_manifest.yaml /opt/nim/etc/default/model_manifest.yaml
COPY ./dependencies.yaml /etc/nim/config/dependencies.yaml

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    nim_dependency_handler


# Build DeepStream server app
FROM nim_builder_base AS build-image

## Copy and untar the DS SDK
COPY ./deepstream_sdk_v7.1.0_x86_64.tbz2 /tmp/${DS_PKG}
RUN tar -xvf /tmp/${DS_PKG} -C / && rm /tmp/${DS_PKG}

RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
        pkg-config \
        libgstrtspserver-1.0-dev \
        libjson-glib-dev \
        libyaml-cpp-dev \
        && \
        rm -rf /var/lib/apt/lists/* && \
        apt autoremove

FROM nim_builder_base AS nim_final

## Copy and untar the DS SDK
#RUN wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/7.0/files?redirect=true&path=deepstream_sdk_v7.0.0_x86_64.tbz2' -O /tmp/${DS_PKG} && tar -xvf /tmp/${DS_PKG} -C / && rm /tmp/${DS_PKG}
COPY ./deepstream_sdk_v7.1.0_x86_64.tbz2 /tmp/${DS_PKG}
RUN tar -xvf /tmp/${DS_PKG} -C / && rm /tmp/${DS_PKG}


WORKDIR $NIM_DIR_PATH

# Patch JPEG Decoder
RUN if [ -e "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so" ]; then \
        mv /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so_bkp; \
    else \
        echo "File 'libgstnvcodec.so' does not exist"; \
    fi
RUN rm -rf /root/.cache/gstreamer-1.0/

## Remove sample, source and install DS
RUN pushd "/opt/nvidia/deepstream/deepstream-7.1" && \
    cp samples/streams/yoga.jpg $NIM_DIR_PATH && \
    popd && \
    /opt/nvidia/deepstream/deepstream/install.sh && \
    rm -rf /tmp/${DS_PKG}

# DALI backend is not used, CVE-2024-5535
RUN rm -rf /opt/tritonserver/backends/dali

RUN \
 ln -s /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
 ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so && \
 ln -s /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec

# IP requirement for removal
RUN dpkg --remove --force-depends libcdparanoia0 libvorbis0a  libvorbisenc2 libvisual-0.4-0 libtwolame0  \
    libtag1v5 libtag1v5-vanilla libspeex1   libvpx7 libjack-jackd2-0 libshout3 libsamplerate0  libopus0 libgl1  libgl1-mesa-dri \
    libglapi-mesa libglvnd0 libglx-mesa0 libglx0 libdrm-amdgpu1 libdrm-common libdrm-intel1 libdrm-nouveau2 libdrm-radeon1  libdrm2 \
    libjbig0 libslang2 libmpg123-0 libmp3lame0 libflac8 libigdgmm12 intel-media-va-driver  i965-va-driver libva-drm2 \
    libavcodec-extra-53 libavcodec58 libavutil56  libchromaprint1 libva2 mesa-va-drivers va-driver-all \
    libavcodec-dev libswresample3 libaom3 libasound2 libasound2-data \
    libass9 libbs2b0 libcodec2-1.0 libdav1d5 libdc1394-25 libdca0 libde265-0 \
    libdecor-0-0 libdvdnav4 libdvdread8 libfaad2 libflite1 libfluidsynth3 \
    libfreeaptx0 libgme0   libgsm1 libgssdp-1.2-0 libgupnp-1.2-1 libgupnp-igd-1.0-4 \
    libilmbase25 libinstpatch-1.0-2 libkate1 liblcms2-2 libldacbt-enc2 liblilv-0-0 libltc11 \
    libmfx1 libmjpegutils-2.1-0 libmodplug1 libmpcdec6   libmpeg2encpp-2.1-0 libmplex2-2.1-0 \
    libnice10 libopenal-data libopenal1 libopenexr25 libopenh264-6 libopenjp2-7 libopenmpt0 \
    libopenni2-0 libqrencode4 librsvg2-2 libsbc1 libsdl2-2.0-0 libserd-0-0 libshine3   \
    libsnappy1v5 libsndio7.0 libsord-0-0 libsoundtouch1 libsoxr0 libspandsp2 libsratom-0-0 \
    libsrt1.4-gnutls libsrtp2-1 libusb-1.0-0 libva-x11-2 libvdpau1 libvo-aacenc0 libvo-amrwbenc0 \
    libvorbisfile3   libwebpmux3 libwebrtc-audio-processing1 libwildmidi2 libx264-163 libx265-199 \
    libxcursor1 libxi6 libxinerama1 libxkbcommon0 libxrandr2 libxss1 libxvidcore4 libzbar0 \
    libzvbi-common libzvbi0 libzxingcore1   ocl-icd-libopencl1 timgm6mb-soundfont x11-common xkb-data \
    gstreamer1.0-gl gstreamer1.0-x poppler-data vdpau-driver-all alsa-topology-conf \
    alsa-ucm-conf fonts-droid-fallback fonts-noto-mono fonts-urw-base35 ghostscript gsfonts \
    imagemagick-6-common libavahi-client3 libavahi-common-data libavahi-common3 libcups2 libdecor-0-plugin-1-cairo \
    libdjvulibre-text libdjvulibre21 libfftw3-double3 libgdk-pixbuf2.0-bin libgl1-amber-dri libgraphene-1.0-0 \
    libgs9 libgs9-common libheif1 libidn12 libijs-0.35 libjbig2dec0 libjxr-tools libjxr0 liblqr-1-0 libltdl7 \
    libmagickcore-6.q16-6 libmagickcore-6.q16-6-extra libmagickwand-6.q16-6 libpaper-utils libpaper1 \
    librsvg2-common libwebpdemux2 libwmflite-0.2-7 libxv1 mesa-vdpau-drivers

RUN dpkg --purge \
    fonts-droid-fallback fonts-noto-mono fonts-urw-base35 gsfonts imagemagick-6-common libpaper1:amd64 poppler-data




RUN rm -f /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstaudioparsers.so \
       /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstx264.so \
       /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstfaad.so \
       /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvoaacenc.so

RUN rm -f /usr/lib/x86_64-linux-gnu/libavresample* /usr/lib/x86_64-linux-gnu/libavutil* \
    /usr/lib/x86_64-linux-gnu/libavcodec* /usr/lib/x86_64-linux-gnu/libavformat* \
    /usr/lib/x86_64-linux-gnu/libavfilter*  /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstde265.so* \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstx265.so* /usr/lib/x86_64-linux-gnu/libde265.so* /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvpx.so*

RUN rm -f /usr/lib/x86_64-linux-gnu/libmpeg2.so* /usr/lib/x86_64-linux-gnu/libmpeg2encpp-2.1.so* /usr/lib/x86_64-linux-gnu/libmpg123.so*  \
    /usr/lib/x86_64-linux-gnu/libx265.so* /usr/lib/x86_64-linux-gnu/libx264.so* /usr/lib/x86_64-linux-gnu/libvpx.so*  \
    /usr/lib/x86_64-linux-gnu/libmpeg2convert.so* /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstopenh264.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so

RUN rm -f \
    /usr/lib/x86_64-linux-gnu/libavresample* /usr/lib/x86_64-linux-gnu/libavutil* \
    /usr/lib/x86_64-linux-gnu/libavcodec* /usr/lib/x86_64-linux-gnu/libavformat* \
    /usr/lib/x86_64-linux-gnu/libavfilter* /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstde265.so* \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstx265.so* /usr/lib/x86_64-linux-gnu/libde265.so* \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvpx.so* \
    /usr/lib/x86_64-linux-gnu/libmpeg2.so.0* /usr/lib/x86_64-linux-gnu/libmpeg2encpp-2.1.so* /usr/lib/x86_64-linux-gnu/libmpg123.so* \
    /usr/lib/x86_64-linux-gnu/libx265.so* /usr/lib/x86_64-linux-gnu/libx264.so* /usr/lib/x86_64-linux-gnu/libvpx.so*   \
    /usr/lib/x86_64-linux-gnu/libmpeg2convert.so* /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstopenh264.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so /usr/lib/x86_64-linux-gnu/gio/modules/libgiolibproxy.so \
    /usr/lib/x86_64-linux-gnu/libdca.so* /usr/lib/x86_64-linux-gnu/libdvdnav.so* \
    /usr/lib/x86_64-linux-gnu/libdvdread.so* /usr/lib/x86_64-linux-gnu/libfaad* \
    /usr/lib/x86_64-linux-gnu/libFLAC.so* /usr/lib/x86_64-linux-gnu/libjbig.so* \
    /usr/lib/x86_64-linux-gnu/libmjpegutils-2.1.so* /usr/lib/x86_64-linux-gnu/libmpeg2encpp-2.1.so* \
    /usr/lib/x86_64-linux-gnu/libmplex2-2.1.so* /usr/lib/x86_64-linux-gnu/libslang.so* \
    /usr/lib/x86_64-linux-gnu/libxvidcore.so* /usr/lib/x86_64-linux-gnu/libzvbi* \
    /usr/lib/x86_64-linux-gnu/libvo-aacenc.so* /usr/lib/x86_64-linux-gnu/libjack* \
    /usr/lib/x86_64-linux-gnu/libswresample.so.3*  && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /tmp/assets && chmod -R 777 /tmp/assets

ADD tao.tgz $NIM_DIR_PATH

RUN groupadd --gid 1000 --non-unique nvs && \
    useradd --create-home --shell /usr/sbin/nologin --uid 1000 --non-unique --gid 1000 nvs && \
    chown -R 1000.1000 $NIM_DIR_PATH && chmod o+w $NIM_DIR_PATH/config* && chmod o+w $NIM_DIR_PATH

USER nvs:1000

## Set Environment variables
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes
ENV NIM_DISABLE_TRITON_STARTUP=1

#  create entrypoint script at location indicated in NIM Playbook
RUN touch $NIM_DIR_PATH/start_server.sh && \
    chmod a+rx $NIM_DIR_PATH/start_server.sh && \
    cat > $NIM_DIR_PATH/start_server.sh <<-EOF
	#!/usr/bin/env bash
	set -eu

	# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
	# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    python3 inference.py
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/opt/nim/start_server.sh"]
CMD []
