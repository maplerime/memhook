#!/usr/bin/env bash
# Launch llama-cli with cudaMalloc redirected to the remote memory server.
#
# usage:
#   ./run.sh           # start server (if needed) + run llama-cli
# env you may override: MEMHOOK_HOST, MEMHOOK_PORT, MEMHOOK_MIN_BYTES, N_TOKENS
set -e

# drop any stale/wrong LD_PRELOAD inherited from the calling shell
unset LD_PRELOAD

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
HOOK="$HERE/libmemhook.so"
if [ ! -f "$HOOK" ]; then
    echo "ERROR: libmemhook.so not found at: $HOOK  (run 'make' in $HERE first)" >&2
    exit 1
fi
echo "using hook: $HOOK" >&2

LLAMA=/root/llama.cpp
MODEL=$LLAMA/models/Qwen2.5-32B-Instruct-Q8_0.gguf

export MEMHOOK_HOST=${MEMHOOK_HOST:-127.0.0.1}
export MEMHOOK_PORT=${MEMHOOK_PORT:-9797}
export MEMHOOK_MIN_BYTES=${MEMHOOK_MIN_BYTES:-1073741824}   # 1 GiB: send big weight buffer to server
export MEMHOOK_MODE=${MEMHOOK_MODE:-zerocopy}              # zerocopy (default) | paged (demand paging)
export MEMHOOK_VERBOSE=${MEMHOOK_VERBOSE:-1}
N_TOKENS=${N_TOKENS:-32}

# zerocopy mode pins host memory for the GPU, which needs a high memlock limit;
# paged mode does not, but raising it here is harmless
ulimit -l unlimited || true

# start the server if nothing is listening on the port
if ! (exec 3<>/dev/tcp/$MEMHOOK_HOST/$MEMHOOK_PORT) 2>/dev/null; then
    echo "starting memserver on port $MEMHOOK_PORT ..."
    "$HERE/memserver" "$MEMHOOK_PORT" &
    echo $! > "$HERE/memserver.pid"
    sleep 1
fi

export LD_PRELOAD="$HOOK"
exec "$LLAMA/build/bin/llama-cli" -m "$MODEL" -ngl 999 \
     -st -p "你好，介绍一下你自己" -n "$N_TOKENS" --no-warmup
