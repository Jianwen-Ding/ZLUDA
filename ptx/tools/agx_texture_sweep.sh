#!/bin/bash
# agx_texture_sweep.sh NAME [agx_texture_check options]: one form through the texture unit and the software sampler, compared
N=$1; shift; D=${OUT_DIR:-texprobe}; mkdir -p $D; T=${AGX_TEXTURE_CHECK:-agx_texture_check}
: ${ZLUDA_AGX_ZOC:?set ZLUDA_AGX_ZOC to zoc}; export ZLUDA_AGX_ZOC
ty=$(ZLUDA_AGX_HARDWARE_TEXTURES=1 ZLUDA_AGX_LOG=1 $T $D/$N.hw "$@" 2>$D/$N.hw.log | sed 's/.*"type":"\(.*\)".*/\1/')
ZLUDA_AGX_SOFTWARE_TEXTURES=1 $T $D/$N.sw "$@" >/dev/null 2>$D/$N.sw.log
sw_note=$(grep -m1 'software texture' $D/$N.hw.log | sed 's/.*software texture (\([^)]*\)).*/ (hw run fell back: \1)/')
echo "$N [$*]: $(python3 $(dirname $0)/agx_texture_compare.py $D/$N.hw $D/$N.sw $ty)$sw_note"
