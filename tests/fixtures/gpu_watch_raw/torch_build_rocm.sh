    rocm*)
        PYTORCH_ROCM_ARCH="gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1200;gfx1201;gfx1150;gfx1151;gfx1250"
        if [[ x ]]; then
            PYTORCH_ROCM_ARCH="gfx900;gfx906;${PYTORCH_ROCM_ARCH}"
        fi
