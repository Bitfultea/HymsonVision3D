#!/bin/bash
python export_model.py
/usr/src/tensorrt/bin/trtexec \
--onnx=../runs/segment/defect_seg_v13/weights/best.onnx \
--saveEngine=hymson3d-seg.engine \
--fp16

# /usr/src/tensorrt/bin/trtexec \
# --onnx=./model/pointpillar_hymson.onnx \
# --fp16 \
# --plugins=build/libpointpillar_core.so \
# --saveEngine=./model/pointpillar_hymson.plan \
# --inputIOFormats=fp16:chw,int32:chw,int32:chw \
# --verbose \
# --dumpLayerInfo \
# --dumpProfile \
# --separateProfileRun \
# --profilingVerbosity=detailed > model/pointpillar_hymson.1017.log 2>&1
