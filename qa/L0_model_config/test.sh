#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CLIENT_LOG="./client.log"
CLIENT=model_config_test.py

SERVER=/opt/tensorrtserver/bin/trtserver
SERVER_TIMEOUT=30
SERVER_LOG_BASE="./inference_server"
source ../common/util.sh

export CUDA_VISIBLE_DEVICES=0

TRIALS="tensorflow_savedmodel tensorflow_graphdef tensorrt_plan caffe2_netdef onnxruntime_onnx pytorch_libtorch custom"

# Copy TensorRT plans into the test model repositories.
for modelpath in \
        autofill_noplatform/tensorrt/bad_input_dims/1 \
        autofill_noplatform/tensorrt/bad_input_type/1 \
        autofill_noplatform/tensorrt/bad_output_dims/1 \
        autofill_noplatform/tensorrt/bad_output_type/1 \
        autofill_noplatform/tensorrt/too_few_inputs/1 \
        autofill_noplatform/tensorrt/too_many_inputs/1 \
        autofill_noplatform/tensorrt/unknown_input/1 \
        autofill_noplatform/tensorrt/unknown_output/1 \
        autofill_noplatform_success/tensorrt/no_name_platform/1 \
        autofill_noplatform_success/tensorrt/empty_config/1     \
        autofill_noplatform_success/tensorrt/no_config/1 ; do
    mkdir -p $modelpath
    cp /data/inferenceserver/qa_model_repository/plan_float32_float32_float32/1/model.plan \
       $modelpath/.
done

rm -f $SERVER_LOG_BASE* $CLIENT_LOG
RET=0

for TRIAL in $TRIALS; do
    # Run all tests that require no autofill but that add the platform to
    # the model config before running the test
    for TARGET in `ls noautofill_platform`; do
        SERVER_ARGS="--model-store=`pwd`/models --strict-model-config=true"
        SERVER_LOG=$SERVER_LOG_BASE.noautofill_platform_${TARGET}.log

        rm -fr models && mkdir models
        cp -r noautofill_platform/$TARGET models/.

        CONFIG=models/$TARGET/config.pbtxt
        EXPECTEDS=models/$TARGET/expected*

        # If there is a config.pbtxt change/add platform to it
        if [ -f $CONFIG ]; then
            sed -i '/platform:/d' $CONFIG
            echo "platform: \"$TRIAL\"" >> $CONFIG
            cat $CONFIG
        fi

        echo -e "Test platform $TRIAL on noautofill_platform/$TARGET" >> $CLIENT_LOG

        # We expect all the tests to fail with one of the expected
        # error messages
        run_server
        if [ "$SERVER_PID" != "0" ]; then
            echo -e "*** FAILED: unexpected success starting $SERVER" >> $CLIENT_LOG
            RET=1
            kill $SERVER_PID
            wait $SERVER_PID
        else
            EXFOUND=0
            for EXPECTED in `ls $EXPECTEDS`; do
                EX=`cat $EXPECTED`
                if grep ^E[0-9][0-9][0-9][0-9].*"$EX" $SERVER_LOG; then
                    echo -e "Found \"$EX\"" >> $CLIENT_LOG
                    EXFOUND=1
                    break
                else
                    echo -e "Not found \"$EX\"" >> $CLIENT_LOG
                fi
            done

            if [ "$EXFOUND" == "0" ]; then
                echo -e "*** FAILED: platform $TRIAL noautofill_platform/$TARGET" >> $CLIENT_LOG
                RET=1
            fi
        fi
    done
done

# Run all autofill tests that don't add a platform to the model config
# before running the test
for TARGET_DIR in `ls -d autofill_noplatform/*/*`; do
    TARGET_DIR_DOT=`echo $TARGET_DIR | tr / .`
    TARGET=`basename ${TARGET_DIR}`

    SERVER_ARGS="--model-store=`pwd`/models --strict-model-config=false"
    SERVER_LOG=$SERVER_LOG_BASE.${TARGET_DIR_DOT}.log

    # If there is a config.pbtxt at the top-level of the test then
    # assume that the directory is a single model. Otherwise assume
    # that the directory is an entire model repository.
    rm -fr models && mkdir models
    if [ -f ${TARGET_DIR}/config.pbtxt ]; then
        cp -r ${TARGET_DIR} models/.
    else
        cp -r ${TARGET_DIR}/* models/.
    fi

    EXPECTEDS=models/$TARGET/expected*

    echo -e "Test ${TARGET_DIR}" >> $CLIENT_LOG

    # We expect all the tests to fail with one of the expected
    # error messages
    run_server
    if [ "$SERVER_PID" != "0" ]; then
        echo -e "*** FAILED: unexpected success starting $SERVER" >> $CLIENT_LOG
        RET=1
        kill $SERVER_PID
        wait $SERVER_PID
    else
        EXFOUND=0
        for EXPECTED in `ls $EXPECTEDS`; do
            EX=`cat $EXPECTED`
            if grep ^E[0-9][0-9][0-9][0-9].*"$EX" $SERVER_LOG; then
                echo -e "Found \"$EX\"" >> $CLIENT_LOG
                EXFOUND=1
                break
            else
                echo -e "Not found \"$EX\"" >> $CLIENT_LOG
            fi
        done

        if [ "$EXFOUND" == "0" ]; then
            echo -e "*** FAILED: ${TARGET_DIR}" >> $CLIENT_LOG
            RET=1
        fi
    fi
done

# Run all autofill tests that are expected to be successful. These
# tests don't add a platform to the model config before running
for TARGET_DIR in `ls -d autofill_noplatform_success/*/*`; do
    TARGET_DIR_DOT=`echo $TARGET_DIR | tr / .`
    TARGET=`basename ${TARGET_DIR}`

    SERVER_ARGS="--model-store=`pwd`/models --strict-model-config=false"
    SERVER_LOG=$SERVER_LOG_BASE.${TARGET_DIR_DOT}.log

    rm -fr models && mkdir models
    cp -r ${TARGET_DIR} models/.

    echo -e "Test $TARGET_DIR" >> $CLIENT_LOG

    set +e

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "*** FAILED: unable to start $SERVER" >> $CLIENT_LOG
        RET=1
    else
        python ./compare_status.py --expected_dir models/$TARGET --model $TARGET >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "*** FAILED: unexpected model config" >> $CLIENT_LOG
            RET=1
        fi

        kill $SERVER_PID
        wait $SERVER_PID
    fi

    set -e
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
    cat $CLIENT_LOG
fi

exit $RET
